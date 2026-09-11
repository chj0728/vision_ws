#include "face_database.h"

#include <algorithm>
#include <memory>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <uuid/uuid.h>

namespace {

// 所有持久化时间使用 UTC，固定格式保留毫秒，避免设备时区变化影响含义。
constexpr const char *kEpochText = "1970-01-01 00:00:00.000";

void executeSql(sqlite3 *db, const std::string &sql) {
  char *error = nullptr;
  if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &error) != SQLITE_OK) {
    const std::string message = error ? error : "unknown SQLite error";
    sqlite3_free(error);
    throw std::runtime_error("[FaceDB] " + message);
  }
}

using Statement = std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)>;

Statement prepareSql(sqlite3 *db, const std::string &sql) {
  sqlite3_stmt *stmt = nullptr;
  if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
    sqlite3_finalize(stmt);
    throw std::runtime_error("[FaceDB] " + std::string(sqlite3_errmsg(db)));
  }
  return Statement(stmt, sqlite3_finalize);
}

// 新库和迁移临时表共用定义，保留主键、自增、默认值和级联删除关系。
std::string createTablesSql(const std::string &persons,
                            const std::string &embeddings) {
  return "CREATE TABLE IF NOT EXISTS " + persons + " ("
         "uuid TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '',"
         "first_seen TEXT NOT NULL DEFAULT '" + kEpochText + "',"
         "last_seen TEXT NOT NULL DEFAULT '" + kEpochText + "',"
         "visit_count INTEGER NOT NULL DEFAULT 1);"
         "CREATE TABLE IF NOT EXISTS " + embeddings + " ("
         "id INTEGER PRIMARY KEY AUTOINCREMENT,"
         "person_uuid TEXT NOT NULL REFERENCES persons(uuid) ON DELETE CASCADE,"
         "embedding BLOB NOT NULL, yaw REAL NOT NULL DEFAULT 0,"
         "pitch REAL NOT NULL DEFAULT 0, face_conf REAL NOT NULL DEFAULT 0,"
         "captured_at TEXT NOT NULL DEFAULT '" + kEpochText + "');";
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// 析构：关闭数据库连接
// ─────────────────────────────────────────────────────────────────────────────
FaceDatabase::~FaceDatabase() {
  if (db_) {
    sqlite3_close(db_);
    db_ = nullptr;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// open()
// ─────────────────────────────────────────────────────────────────────────────
int FaceDatabase::open(const std::string &db_path) {
  std::lock_guard<std::mutex> lock(mu_);
  if (db_) {
    sqlite3_close(db_);
    db_ = nullptr;
    persons_.clear();
  }

  const int rc = sqlite3_open(db_path.c_str(), &db_);
  if (rc != SQLITE_OK) {
    std::fprintf(stderr, "[FaceDB] cannot open %s: %s\n", db_path.c_str(),
                 sqlite3_errmsg(db_));
    sqlite3_close(db_);
    db_ = nullptr;
    return -1;
  }

  // WAL 模式改善并发读取；外键保证删除人物时同步删除特征。
  sqlite3_exec(db_, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
  sqlite3_exec(db_, "PRAGMA foreign_keys=ON;", nullptr, nullptr, nullptr);

  try {
    initSchema();
    migrateTimeColumns();
    loadFromDB();
  } catch (const std::exception &error) {
    // 迁移失败时不暴露半初始化的数据库对象，调用方仍通过 -1 判断失败。
    std::fprintf(stderr, "[FaceDB] initialization failed: %s\n", error.what());
    sqlite3_close(db_);
    db_ = nullptr;
    persons_.clear();
    return -1;
  }

  std::printf("[FaceDB] opened %s  persons=%d\n", db_path.c_str(),
              static_cast<int>(persons_.size()));
  return static_cast<int>(persons_.size());
}

// ─────────────────────────────────────────────────────────────────────────────
// 初始化表结构及旧库时间列迁移
// ─────────────────────────────────────────────────────────────────────────────
void FaceDatabase::initSchema() {
  executeSql(db_, createTablesSql("persons", "face_embeddings"));
  executeSql(db_, "CREATE INDEX IF NOT EXISTS idx_emb_person ON "
                  "face_embeddings(person_uuid);");
}

void FaceDatabase::migrateTimeColumns() {
  // 使用实际列声明识别旧库，不依赖可被其他组件使用的 user_version。
  int integer_time_columns = 0;
  const std::vector<std::pair<std::string, std::vector<std::string>>> tables = {
      {"persons", {"uuid", "name", "first_seen", "last_seen", "visit_count"}},
      {"face_embeddings", {"id", "person_uuid", "embedding", "yaw", "pitch",
                           "face_conf", "captured_at"}}};
  for (const auto &[table, columns] : tables) {
    auto stmt = prepareSql(db_, "PRAGMA table_info(" + table + ");");
    std::size_t index = 0;
    int rc;
    while ((rc = sqlite3_step(stmt.get())) == SQLITE_ROW) {
      const std::string name = reinterpret_cast<const char *>(
          sqlite3_column_text(stmt.get(), 1));
      if (index >= columns.size() || name != columns[index++]) {
        throw std::runtime_error("[FaceDB] unsupported table layout: " + table);
      }
      if (name == "first_seen" || name == "last_seen" || name == "captured_at") {
        const std::string type = reinterpret_cast<const char *>(
            sqlite3_column_text(stmt.get(), 2));
        if (type == "INTEGER") {
          ++integer_time_columns;
        } else if (type != "TEXT") {
          throw std::runtime_error("[FaceDB] unsupported time column type");
        }
      }
    }
    if (rc != SQLITE_DONE || index != columns.size()) {
      throw std::runtime_error("[FaceDB] cannot inspect table: " + table);
    }
  }
  if (integer_time_columns == 0) {
    return;
  }

  // 关闭外键后再开始事务，避免重建父表时触发旧库特征的级联删除。
  executeSql(db_, "PRAGMA foreign_keys=OFF;");
  try {
    executeSql(db_, "BEGIN IMMEDIATE;");
    std::vector<std::string> schema_objects;
    {
      auto stmt = prepareSql(db_,
          "SELECT sql FROM sqlite_master WHERE type IN ('index','trigger') "
          "AND tbl_name IN ('persons','face_embeddings') AND sql IS NOT NULL;");
      int rc;
      while ((rc = sqlite3_step(stmt.get())) == SQLITE_ROW) {
        schema_objects.emplace_back(reinterpret_cast<const char *>(
            sqlite3_column_text(stmt.get(), 0)));
      }
      if (rc != SQLITE_DONE) {
        throw std::runtime_error("[FaceDB] cannot read indexes and triggers");
      }
    }
    // 即使历史最高 ID 对应的行已删除，也保留 AUTOINCREMENT 高水位。
    executeSql(db_, "CREATE TEMP TABLE face_time_sequence AS SELECT seq FROM "
                    "sqlite_sequence WHERE name='face_embeddings';");
    // 不复用已有同名表，防止覆盖非本次迁移的数据。
    {
      auto stmt = prepareSql(db_, "SELECT 1 FROM sqlite_master WHERE name IN "
          "('persons_time_new','face_embeddings_time_new');");
      if (sqlite3_step(stmt.get()) != SQLITE_DONE) {
        throw std::runtime_error("[FaceDB] time migration table name conflict");
      }
    }
    executeSql(db_, createTablesSql("persons_time_new", "face_embeddings_time_new"));
    const auto time_expression = [](const std::string &column) {
      // 整数按 Unix 毫秒转换；已有 TEXT 时间在部分迁移场景中原样保留。
      return "CASE WHEN typeof(" + column + ")='integer' THEN "
             "strftime('%Y-%m-%d %H:%M:%f'," + column + "/1000.0,'unixepoch') "
             "ELSE " + column + " END";
    };
    executeSql(db_, "INSERT INTO persons_time_new SELECT uuid,name," +
        time_expression("first_seen") + "," + time_expression("last_seen") +
        ",visit_count FROM persons;");
    executeSql(db_, "INSERT INTO face_embeddings_time_new SELECT "
        "id,person_uuid,embedding,yaw,pitch,face_conf," +
        time_expression("captured_at") + " FROM face_embeddings;");
    {
      auto stmt = prepareSql(db_,
          "SELECT 1 FROM (SELECT first_seen AS t FROM persons_time_new "
          "UNION ALL SELECT last_seen FROM persons_time_new "
          "UNION ALL SELECT captured_at FROM face_embeddings_time_new) "
          "WHERE typeof(t)!='text' OR length(t)!=23 OR "
          "strftime('%Y-%m-%d %H:%M:%f',t) IS NULL OR "
          "strftime('%Y-%m-%d %H:%M:%f',t)!=t LIMIT 1;");
      if (sqlite3_step(stmt.get()) != SQLITE_DONE) {
        throw std::runtime_error("[FaceDB] invalid timestamp during migration");
      }
    }
    executeSql(db_, "DROP TABLE face_embeddings; DROP TABLE persons;"
        "ALTER TABLE persons_time_new RENAME TO persons;"
        "ALTER TABLE face_embeddings_time_new RENAME TO face_embeddings;");
    for (const auto &sql : schema_objects) {
      executeSql(db_, sql);
    }
    executeSql(db_, "UPDATE sqlite_sequence SET seq=MAX(seq,COALESCE("
        "(SELECT MAX(seq) FROM face_time_sequence),0)) "
        "WHERE name='face_embeddings'; DROP TABLE face_time_sequence;");
    {
      auto stmt = prepareSql(db_, "PRAGMA foreign_key_check;");
      if (sqlite3_step(stmt.get()) != SQLITE_DONE) {
        throw std::runtime_error("[FaceDB] foreign key check failed after migration");
      }
    }
    executeSql(db_, "COMMIT;");
  } catch (...) {
    sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
    sqlite3_exec(db_, "PRAGMA foreign_keys=ON;", nullptr, nullptr, nullptr);
    throw;
  }
  executeSql(db_, "PRAGMA foreign_keys=ON;");
}

// ─────────────────────────────────────────────────────────────────────────────
// 加载人物和有效特征到内存，时间文本转换为原有毫秒接口
// ─────────────────────────────────────────────────────────────────────────────
void FaceDatabase::loadFromDB() {
  persons_.clear();

  // 第一步：读取人物信息，将 UTC 文本还原为 Unix 毫秒。
  {
    sqlite3_stmt *stmt = nullptr;
    const char *sql =
        "SELECT uuid, name, "
        "CAST(strftime('%s',first_seen) AS INTEGER)*1000 + "
        "CAST(substr(first_seen,21,3) AS INTEGER), "
        "CAST(strftime('%s',last_seen) AS INTEGER)*1000 + "
        "CAST(substr(last_seen,21,3) AS INTEGER), visit_count FROM persons;";
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
      return;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
      PersonRecord rec;
      rec.uuid = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0));
      rec.name = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
      rec.first_seen = sqlite3_column_int64(stmt, 2);
      rec.last_seen = sqlite3_column_int64(stmt, 3);
      rec.visit_count = sqlite3_column_int(stmt, 4);
      persons_.push_back(std::move(rec));
    }
    sqlite3_finalize(stmt);
  }

  // 第二步：一次查询全部特征，按特征 ID 顺序归入对应人物。
  {
    sqlite3_stmt *stmt = nullptr;
    const char *sql =
        "SELECT person_uuid, embedding FROM face_embeddings ORDER BY id ASC;";
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
      return;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
      const std::string uuid =
          reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0));
      const void *blob = sqlite3_column_blob(stmt, 1);
      const int bytes = sqlite3_column_bytes(stmt, 1);

      if (bytes != static_cast<int>(FaceEmbedding::DIM * sizeof(float)))
        continue;

      // 按 UUID 查找特征所属人物。
      auto it =
          std::find_if(persons_.begin(), persons_.end(),
                       [&](const PersonRecord &p) { return p.uuid == uuid; });
      if (it == persons_.end())
        continue;

      FaceEmbedding emb{};
      std::memcpy(emb.v, blob, FaceEmbedding::DIM * sizeof(float));
      if (!ArcFaceTRT::l2Normalize(emb))
        continue;
      it->embeddings.push_back(emb);
    }
    sqlite3_finalize(stmt);
  }

  // 从内存中排除没有有效特征的人物，不删除数据库原始记录。
  persons_.erase(std::remove_if(persons_.begin(), persons_.end(),
                                [](const PersonRecord &p) {
                                  return p.embeddings.empty();
                                }),
                 persons_.end());
}

// ─────────────────────────────────────────────────────────────────────────────
// identify()
// ─────────────────────────────────────────────────────────────────────────────
FaceMatch FaceDatabase::identify(const FaceEmbedding &query,
                                 float threshold) const {
  std::lock_guard<std::mutex> lock(mu_);

  FaceMatch best;
  FaceEmbedding normalized_query = query;
  if (!ArcFaceTRT::l2Normalize(normalized_query))
    return best;
  for (const auto &person : persons_) {
    for (const auto &emb : person.embeddings) {
      const float sim = ArcFaceTRT::cosineSim(normalized_query, emb);
      if (sim > best.similarity) {
        best.similarity = sim;
        best.uuid = person.uuid;
        best.name = person.name;
      }
    }
  }

  if (best.similarity >= threshold) {
    best.identified = true;
  } else {
    best.uuid = "";
    best.name = "";
    best.identified = false;
  }
  return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// registerPerson()
// ─────────────────────────────────────────────────────────────────────────────
std::string FaceDatabase::registerPerson(
    const std::string &name, const std::vector<FaceEmbedding> &embeddings,
    const std::vector<float> &yaws, const std::vector<float> &pitches,
    const std::vector<float> &confs) {
  if (embeddings.empty())
    return "";

  struct ValidEmbedding {
    FaceEmbedding embedding;
    size_t source_index;
  };
  std::vector<ValidEmbedding> valid_embeddings;
  const size_t input_count =
      std::min(embeddings.size(), static_cast<size_t>(kMaxEmbeddings));
  valid_embeddings.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    FaceEmbedding normalized = embeddings[i];
    if (ArcFaceTRT::l2Normalize(normalized))
      valid_embeddings.push_back({normalized, i});
  }
  if (valid_embeddings.empty())
    return "";

  std::lock_guard<std::mutex> lock(mu_);
  if (!db_)
    return "";

  const std::string uuid = generateUUID();
  const int64_t now = nowMs();

  // 只保存已经归一化成功的特征。
  const int n = static_cast<int>(valid_embeddings.size());

  // 一次注册中，人物与全部特征共用事务和同一个毫秒时间。
  sqlite3_exec(db_, "BEGIN;", nullptr, nullptr, nullptr);

  // 插入人物；绑定毫秒值，在 SQL 中转换为 UTC 日期时间文本。
  {
    sqlite3_stmt *stmt = nullptr;
    const char *sql =
        "INSERT INTO persons(uuid, name, first_seen, last_seen, visit_count) "
        "VALUES(?,?,strftime('%Y-%m-%d %H:%M:%f',?/1000.0,'unixepoch'),"
        "strftime('%Y-%m-%d %H:%M:%f',?/1000.0,'unixepoch'),1);";
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
      sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
      return "";
    }
    sqlite3_bind_text(stmt, 1, uuid.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 3, now);
    sqlite3_bind_int64(stmt, 4, now);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
  }

  // 插入特征；captured_at 使用本次注册时间。
  PersonRecord rec;
  rec.uuid = uuid;
  rec.name = name;
  rec.first_seen = now;
  rec.last_seen = now;
  rec.visit_count = 1;

  for (int i = 0; i < n; ++i) {
    const auto &valid = valid_embeddings[static_cast<size_t>(i)];
    const FaceEmbedding &emb = valid.embedding;
    const size_t source_index = valid.source_index;

    const float yaw = source_index < yaws.size() ? yaws[source_index] : 0.f;
    const float pitch =
        source_index < pitches.size() ? pitches[source_index] : 0.f;
    const float face_conf =
        source_index < confs.size() ? confs[source_index] : 0.f;

    sqlite3_stmt *stmt = nullptr;
    const char *sql = "INSERT INTO face_embeddings(person_uuid, embedding, "
                      "yaw, pitch, face_conf, captured_at) "
                      "VALUES(?,?,?,?,?,strftime('%Y-%m-%d %H:%M:%f',?/1000.0,'unixepoch'));";
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) == SQLITE_OK) {
      sqlite3_bind_text(stmt, 1, uuid.c_str(), -1, SQLITE_STATIC);
      sqlite3_bind_blob(stmt, 2, emb.v, FaceEmbedding::DIM * sizeof(float),
                        SQLITE_STATIC);
      sqlite3_bind_double(stmt, 3, static_cast<double>(yaw));
      sqlite3_bind_double(stmt, 4, static_cast<double>(pitch));
      sqlite3_bind_double(stmt, 5, static_cast<double>(face_conf));
      sqlite3_bind_int64(stmt, 6, now);
      sqlite3_step(stmt);
      sqlite3_finalize(stmt);
    }
    rec.embeddings.push_back(emb);
  }

  sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr);
  persons_.push_back(std::move(rec));

  std::printf("[FaceDB] registered uuid=%s name='%s' embeddings=%d\n",
              uuid.substr(0, 8).c_str(), name.c_str(), n);
  return uuid;
}

// ─────────────────────────────────────────────────────────────────────────────
// updateName()
// ─────────────────────────────────────────────────────────────────────────────
bool FaceDatabase::updateName(const std::string &uuid,
                              const std::string &name) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!db_)
    return false;

  sqlite3_stmt *stmt = nullptr;
  const char *sql = "UPDATE persons SET name=? WHERE uuid=?;";
  if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
    return false;
  sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC);
  sqlite3_bind_text(stmt, 2, uuid.c_str(), -1, SQLITE_STATIC);
  const int rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);

  if (rc != SQLITE_DONE)
    return false;
  if (sqlite3_changes(db_) == 0)
    return false;

  // 数据库更新成功后同步内存姓名。
  for (auto &p : persons_) {
    if (p.uuid == uuid) {
      p.name = name;
      break;
    }
  }
  std::printf("[FaceDB] updated name: uuid=%s  name='%s'\n",
              uuid.substr(0, 8).c_str(), name.c_str());
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// touchPerson()
// ─────────────────────────────────────────────────────────────────────────────
void FaceDatabase::touchPerson(const std::string &uuid) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!db_)
    return;
  const int64_t now = nowMs();
  sqlite3_stmt *stmt = nullptr;
  const char *sql =
      "UPDATE persons SET last_seen="
      "strftime('%Y-%m-%d %H:%M:%f',?/1000.0,'unixepoch'), "
      "visit_count=visit_count+1 WHERE uuid=?;";
  if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
    return;
  sqlite3_bind_int64(stmt, 1, now);
  sqlite3_bind_text(stmt, 2, uuid.c_str(), -1, SQLITE_STATIC);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  for (auto &p : persons_) {
    if (p.uuid == uuid) {
      p.last_seen = now;
      p.visit_count++;
      break;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// personCount()
// ─────────────────────────────────────────────────────────────────────────────
int FaceDatabase::personCount() const {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<int>(persons_.size());
}

// ─────────────────────────────────────────────────────────────────────────────
// 使用 libuuid 生成人物 UUID v4
// ─────────────────────────────────────────────────────────────────────────────
std::string FaceDatabase::generateUUID() const {
  uuid_t id;
  uuid_generate_random(id);
  char s[37];
  uuid_unparse_lower(id, s);
  return std::string(s);
}

// ─────────────────────────────────────────────────────────────────────────────
// nowMs()
// ─────────────────────────────────────────────────────────────────────────────
int64_t FaceDatabase::nowMs() const {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
      .count();
}
