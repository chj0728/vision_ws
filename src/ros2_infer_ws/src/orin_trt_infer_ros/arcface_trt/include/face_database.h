#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <sqlite3.h>

#include "arcface_trt.h"  // FaceEmbedding, ArcFaceTRT::cosineSim / l2Normalize

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

/** In-memory record for one registered person. */
struct PersonRecord {
    std::string uuid;                         // UUID v4 (primary key)
    std::string name;                         // human-readable name; "" = unnamed
    std::vector<FaceEmbedding> embeddings;    // 1-kMaxEmbeddings per person
    int64_t first_seen{0};                    // unix ms
    int64_t last_seen{0};
    int32_t visit_count{1};
};

/** Result returned by FaceDatabase::identify(). */
struct FaceMatch {
    std::string uuid;          // matched UUID, or "" if not identified
    std::string name;          // matched name, or ""
    float       similarity{0.f};
    bool        identified{false};
};

// ─────────────────────────────────────────────────────────────────────────────
// FaceDatabase
//
// SQLite-backed face gallery.  Loads all embeddings into RAM on open() for
// fast linear-scan identification.  All write operations (register, updateName)
// are write-through: SQLite is updated immediately, then the in-memory mirror.
//
// Schema (auto-created on first open):
//   persons(uuid TEXT PK, name TEXT, first_seen INTEGER, last_seen INTEGER,
//           visit_count INTEGER)
//   face_embeddings(id INTEGER PK AUTOINCREMENT, person_uuid TEXT FK,
//                   embedding BLOB, yaw REAL, pitch REAL, face_conf REAL,
//                   captured_at INTEGER)
//
// Thread-safety: internal mutex guards all public methods.
// ─────────────────────────────────────────────────────────────────────────────
class FaceDatabase {
public:
    static constexpr int kMaxEmbeddings = 5;  // max embeddings stored per person

    FaceDatabase() = default;
    ~FaceDatabase();

    FaceDatabase(const FaceDatabase&)            = delete;
    FaceDatabase& operator=(const FaceDatabase&) = delete;

    /** Open (or create) the SQLite file at db_path.
     *  Returns number of persons loaded, or -1 on error. */
    int open(const std::string& db_path);

    /** Identify a query embedding against all persons in the gallery.
     *  Uses max-cosine-similarity across each person's embeddings.
     *  Returns identified=true only if best similarity >= threshold. */
    FaceMatch identify(const FaceEmbedding& query, float threshold) const;

    /** Register a new person with up to kMaxEmbeddings embeddings.
     *  yaw/pitch/face_conf are optional metadata for each embedding (pass 0 if unavailable).
     *  Returns the new UUID v4, or "" on error. */
    std::string registerPerson(const std::string& name,
                               const std::vector<FaceEmbedding>& embeddings,
                               const std::vector<float>& yaws   = {},
                               const std::vector<float>& pitches = {},
                               const std::vector<float>& confs   = {});

    /** Update the person's name by UUID (write-through to SQLite). */
    bool updateName(const std::string& uuid, const std::string& name);

    /** Update last_seen and increment visit_count for a person. */
    void touchPerson(const std::string& uuid);

    int  personCount() const;
    bool isOpen()      const { return db_ != nullptr; }

private:
    void initSchema();
    void loadFromDB();
    std::string generateUUID() const;
    int64_t nowMs() const;
    bool execSQL(const std::string& sql, std::string* err = nullptr);

    mutable std::mutex mu_;
    sqlite3* db_{nullptr};
    std::vector<PersonRecord> persons_;  // in-memory mirror
};
