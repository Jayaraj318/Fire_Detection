from sqlalchemy import create_engine, text
DATABASE_URL = 'postgresql://neondb_owner:npg_68qSGaezWxwu@ep-green-star-adughbxb-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require'
engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    print("Checking/Adding status column...")
    try:
        conn.execute(text("ALTER TABLE detections ADD COLUMN status VARCHAR DEFAULT 'active'"))
        conn.commit()
        print("✅ Added status column")
    except Exception as e:
        print(f"ℹ️ Status column not added (likely exists): {e}")

    print("Checking/Adding severity column...")
    try:
        conn.execute(text("ALTER TABLE detections ADD COLUMN severity VARCHAR DEFAULT 'normal'"))
        conn.commit()
        print("✅ Added severity column")
    except Exception as e:
        print(f"ℹ️ Severity column not added (likely exists): {e}")
