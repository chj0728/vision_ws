#include "face_database.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <uuid/uuid.h>

// ─────────────────────────────────────────────────────────────────────────────
// Destructor
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
int FaceDatabase::open(const std::string& db_path) {
    std::lock_guard<std::mutex> lock(mu_);
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
        persons_.clear();
    }

    const int rc = sqlite3_open(db_path.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::fprintf(stderr, "[FaceDB] cannot open %s: %s\n",
                     db_path.c_str(), sqlite3_errmsg(db_));
        sqlite3_close(db_);
        db_ = nullptr;
        return -1;
    }

    // WAL mode: better concurrent read performance
    sqlite3_exec(db_, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
    sqlite3_exec(db_, "PRAGMA foreign_keys=ON;",  nullptr, nullptr, nullptr);

    initSchema();
    loadFromDB();

    std::printf("[FaceDB] opened %s  persons=%d\n",
                db_path.c_str(), static_cast<int>(persons_.size()));
    return static_cast<int>(persons_.size());
}

// ─────────────────────────────────────────────────────────────────────────────
// initSchema()  — creates tables if they don't exist
// ─────────────────────────────────────────────────────────────────────────────
void FaceDatabase::initSchema() {
    const char* sql =
        "CREATE TABLE IF NOT EXISTS persons ("
        "  uuid        TEXT    PRIMARY KEY,"
        "  name        TEXT    NOT NULL DEFAULT '',"
        "  first_seen  INTEGER NOT NULL DEFAULT 0,"
        "  last_seen   INTEGER NOT NULL DEFAULT 0,"
        "  visit_count INTEGER NOT NULL DEFAULT 1"
        ");"
        "CREATE TABLE IF NOT EXISTS face_embeddings ("
        "  id          INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  person_uuid TEXT    NOT NULL REFERENCES persons(uuid) ON DELETE CASCADE,"
        "  embedding   BLOB    NOT NULL,"
        "  yaw         REAL    NOT NULL DEFAULT 0,"
        "  pitch       REAL    NOT NULL DEFAULT 0,"
        "  face_conf   REAL    NOT NULL DEFAULT 0,"
        "  captured_at INTEGER NOT NULL DEFAULT 0"
        ");"
        "CREATE INDEX IF NOT EXISTS idx_emb_person ON face_embeddings(person_uuid);";

    char* err = nullptr;
    if (sqlite3_exec(db_, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        std::string msg = err ? err : "(null)";
        sqlite3_free(err);
        throw std::runtime_error("[FaceDB] initSchema failed: " + msg);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// loadFromDB()  — loads all persons + embeddings into in-memory gallery
// ─────────────────────────────────────────────────────────────────────────────
void FaceDatabase::loadFromDB() {
    persons_.clear();

    // Step 1: load persons
    {
        sqlite3_stmt* stmt = nullptr;
        const char* sql = "SELECT uuid, name, first_seen, last_seen, visit_count FROM persons;";
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            PersonRecord rec;
            rec.uuid        = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            rec.name        = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            rec.first_seen  = sqlite3_column_int64(stmt, 2);
            rec.last_seen   = sqlite3_column_int64(stmt, 3);
            rec.visit_count = sqlite3_column_int(stmt,  4);
            persons_.push_back(std::move(rec));
        }
        sqlite3_finalize(stmt);
    }

    // Step 2: load embeddings (one query per person to keep order clear)
    {
        sqlite3_stmt* stmt = nullptr;
        const char* sql =
            "SELECT person_uuid, embedding FROM face_embeddings ORDER BY id ASC;";
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return;

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const std::string uuid =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            const void* blob   = sqlite3_column_blob(stmt, 1);
            const int   bytes  = sqlite3_column_bytes(stmt, 1);

            if (bytes != static_cast<int>(FaceEmbedding::DIM * sizeof(float))) continue;

            // Find matching person
            auto it = std::find_if(persons_.begin(), persons_.end(),
                                   [&](const PersonRecord& p) { return p.uuid == uuid; });
            if (it == persons_.end()) continue;

            FaceEmbedding emb;
            std::memcpy(emb.v, blob, FaceEmbedding::DIM * sizeof(float));
            ArcFaceTRT::l2Normalize(emb);
            it->embeddings.push_back(emb);
        }
        sqlite3_finalize(stmt);
    }

    // Remove persons with no embeddings (shouldn't normally happen)
    persons_.erase(
        std::remove_if(persons_.begin(), persons_.end(),
                       [](const PersonRecord& p) { return p.embeddings.empty(); }),
        persons_.end());
}

// ─────────────────────────────────────────────────────────────────────────────
// identify()
// ─────────────────────────────────────────────────────────────────────────────
FaceMatch FaceDatabase::identify(const FaceEmbedding& query, float threshold) const {
    std::lock_guard<std::mutex> lock(mu_);

    FaceMatch best;
    for (const auto& person : persons_) {
        for (const auto& emb : person.embeddings) {
            const float sim = ArcFaceTRT::cosineSim(query, emb);
            if (sim > best.similarity) {
                best.similarity = sim;
                best.uuid       = person.uuid;
                best.name       = person.name;
            }
        }
    }

    if (best.similarity >= threshold) {
        best.identified = true;
    } else {
        best.uuid       = "";
        best.name       = "";
        best.identified = false;
    }
    return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// registerPerson()
// ─────────────────────────────────────────────────────────────────────────────
std::string FaceDatabase::registerPerson(const std::string& name,
                                         const std::vector<FaceEmbedding>& embeddings,
                                         const std::vector<float>& yaws,
                                         const std::vector<float>& pitches,
                                         const std::vector<float>& confs) {
    if (embeddings.empty()) return "";
    std::lock_guard<std::mutex> lock(mu_);
    if (!db_) return "";

    const std::string uuid = generateUUID();
    const int64_t now      = nowMs();

    // Limit stored embeddings
    const int n = std::min(static_cast<int>(embeddings.size()), kMaxEmbeddings);

    // Begin transaction
    sqlite3_exec(db_, "BEGIN;", nullptr, nullptr, nullptr);

    // Insert person
    {
        sqlite3_stmt* stmt = nullptr;
        const char* sql =
            "INSERT INTO persons(uuid, name, first_seen, last_seen, visit_count) "
            "VALUES(?,?,?,?,1);";
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

    // Insert embeddings
    PersonRecord rec;
    rec.uuid        = uuid;
    rec.name        = name;
    rec.first_seen  = now;
    rec.last_seen   = now;
    rec.visit_count = 1;

    for (int i = 0; i < n; ++i) {
        FaceEmbedding emb = embeddings[static_cast<size_t>(i)];
        ArcFaceTRT::l2Normalize(emb);

        const float yaw      = (i < static_cast<int>(yaws.size()))    ? yaws[static_cast<size_t>(i)]    : 0.f;
        const float pitch    = (i < static_cast<int>(pitches.size())) ? pitches[static_cast<size_t>(i)] : 0.f;
        const float face_conf= (i < static_cast<int>(confs.size()))   ? confs[static_cast<size_t>(i)]   : 0.f;

        sqlite3_stmt* stmt = nullptr;
        const char* sql =
            "INSERT INTO face_embeddings(person_uuid, embedding, yaw, pitch, face_conf, captured_at) "
            "VALUES(?,?,?,?,?,?);";
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, uuid.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_blob(stmt, 2, emb.v, FaceEmbedding::DIM * sizeof(float), SQLITE_STATIC);
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
bool FaceDatabase::updateName(const std::string& uuid, const std::string& name) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!db_) return false;

    sqlite3_stmt* stmt = nullptr;
    const char* sql = "UPDATE persons SET name=? WHERE uuid=?;";
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, uuid.c_str(), -1, SQLITE_STATIC);
    const int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) return false;
    if (sqlite3_changes(db_) == 0) return false;

    // Update in-memory mirror
    for (auto& p : persons_) {
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
void FaceDatabase::touchPerson(const std::string& uuid) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!db_) return;
    const int64_t now = nowMs();
    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "UPDATE persons SET last_seen=?, visit_count=visit_count+1 WHERE uuid=?;";
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return;
    sqlite3_bind_int64(stmt, 1, now);
    sqlite3_bind_text(stmt, 2, uuid.c_str(), -1, SQLITE_STATIC);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    for (auto& p : persons_) {
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
// generateUUID()  — libuuid UUID v4
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
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}
