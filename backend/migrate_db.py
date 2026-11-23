import sqlite3

def run_migration():
    conn = sqlite3.connect('health_ai.db')
    cur = conn.cursor()

    def add_col(table, col, col_type):
        cur.execute(f"PRAGMA table_info({table})")
        cols = [c[1] for c in cur.fetchall()]
        if col not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            print(f"Added {col} to {table}")
        else:
            print(f"Column {col} already exists in {table}")

    add_col('prescriptions', 'tests_advised', "TEXT DEFAULT ''")
    add_col('prescriptions', 'follow_up', "VARCHAR(100) DEFAULT ''")
    add_col('medicines', 'timing', "VARCHAR(50) DEFAULT ''")
    add_col('medicines', 'meal_instruction', "VARCHAR(50) DEFAULT ''")
    add_col('medicine_schedules', 'prescription_id', "INTEGER DEFAULT NULL")
    add_col('medicine_schedules', 'frequency', "VARCHAR(50) DEFAULT ''")
    add_col('medicine_schedules', 'reminder_type', "VARCHAR(30) DEFAULT 'medicine'")
    add_col('ml_assessments', 'source_type', "VARCHAR(30) DEFAULT 'manual'")

    conn.commit()

    for table in ['prescriptions', 'medicines', 'medicine_schedules', 'ml_assessments']:
        cur.execute(f"PRAGMA table_info({table})")
        print(table, "columns:", [c[1] for c in cur.fetchall()])

    conn.close()
    print("Database columns synchronized successfully.")

if __name__ == '__main__':
    run_migration()
