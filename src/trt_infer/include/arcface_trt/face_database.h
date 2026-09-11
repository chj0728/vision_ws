#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <sqlite3.h>

#include "arcface_trt.h" // FaceEmbedding, ArcFaceTRT::cosineSim / l2Normalize

/** @brief 内存中的人物记录；保留毫秒时间戳类型，兼容已有调用代码。 */
struct PersonRecord {
  std::string uuid;                     // 人物 UUID v4，数据库主键
  std::string name;                     // 人物姓名，空字符串表示尚未命名
  std::vector<FaceEmbedding> embeddings; // 每个人最多 kMaxEmbeddings 条特征
  int64_t first_seen{0};                // 首次注册时间，Unix 毫秒；读库时由 UTC 文本转换
  int64_t last_seen{0};                 // 最近一次出现时间，Unix 毫秒
  int32_t visit_count{1};               // 注册为 1，每次 touchPerson 调用递增
};

/** @brief 人脸查询结果；识别失败时 UUID 和姓名为空。 */
struct FaceMatch {
  std::string uuid;       // 匹配人物的 UUID
  std::string name;       // 匹配人物的姓名，可为空
  float similarity{0.f}; // 最佳余弦相似度
  bool identified{false}; // 最佳相似度是否达到查询阈值
};

/**
 * @brief SQLite 人脸库，打开时把有效特征加载到内存供快速比较。
 *
 * persons.first_seen/last_seen 和 face_embeddings.captured_at 使用 TEXT，
 * 固定为 UTC 的 YYYY-MM-DD HH:MM:SS.SSS；内存记录仍使用 Unix 毫秒。
 * open() 自动事务迁移旧 INTEGER 毫秒字段，失败时回滚并返回 -1。
 * 数据库操作由内部互斥锁保护；isOpen() 仅查询句柄，不能与 open() 并发调用。
 */
class FaceDatabase {
public:
  static constexpr int kMaxEmbeddings = 5; // 每个人最多保存的特征数量

  FaceDatabase() = default;
  ~FaceDatabase();

  FaceDatabase(const FaceDatabase &) = delete;
  FaceDatabase &operator=(const FaceDatabase &) = delete;

  /** @brief 打开或创建人脸库，自动迁移旧时间列；成功返回加载人数，失败返回 -1。 */
  int open(const std::string &db_path);

  /** @brief 取查询特征与全库特征的最大余弦相似度，达到 threshold 才识别成功。 */
  FaceMatch identify(const FaceEmbedding &query, float threshold) const;

  /**
   * @brief 注册新人物，返回 UUID；没有有效特征或失败时返回空字符串。
   * @param embeddings 最多处理前 kMaxEmbeddings 条输入，忽略不能归一化的特征。
   * @param yaws 与输入特征对应的偏航角，缺省位置使用 0。
   * @param pitches 与输入特征对应的俯仰角，缺省位置使用 0。
   * @param confs 与输入特征对应的人脸置信度，缺省位置使用 0。
   * 三个时间列使用本次注册的同一时刻；captured_at 当前表示入库时间。
   */
  std::string registerPerson(const std::string &name,
                             const std::vector<FaceEmbedding> &embeddings,
                             const std::vector<float> &yaws = {},
                             const std::vector<float> &pitches = {},
                             const std::vector<float> &confs = {});

  /** @brief 按 UUID 更新数据库姓名，成功后同步内存记录。 */
  bool updateName(const std::string &uuid, const std::string &name);

  /** @brief 更新最近出现时间并累计访问次数，首次注册时间不变。 */
  void touchPerson(const std::string &uuid);

  /** @brief 返回内存中包含有效特征的人物数量。 */
  int personCount() const;
  bool isOpen() const { return db_ != nullptr; }

private:
  /** @brief 为新库创建 TEXT 时间列及人脸特征索引。 */
  void initSchema();
  /** @brief 事务重建旧时间列，保留特征、人物、索引、触发器和自增序号。 */
  void migrateTimeColumns();
  /** @brief 加载有效特征，将 UTC 时间文本转换回内存 Unix 毫秒。 */
  void loadFromDB();
  std::string generateUUID() const;
  /** @brief 获取当前 Unix 毫秒，SQL 写入时统一格式化为 UTC 文本。 */
  int64_t nowMs() const;

  mutable std::mutex mu_;
  sqlite3 *db_{nullptr};
  std::vector<PersonRecord> persons_; // 内存人脸库，不包含无有效特征的人物
};
