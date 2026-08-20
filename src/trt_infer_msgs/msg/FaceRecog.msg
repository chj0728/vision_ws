# ── 人脸识别 (ArcFace)
string       person_uuid         # 人物 UUID v4；""表示未识别或库中无此人
string       person_name         # 数据库中存储的姓名；""表示尚未命名
float32      face_recog_conf     # 余弦相似度 [0,1]；0 表示未识别或未开启
float32[512] face_embedding      # L2 归一化特征向量；全零表示未提取