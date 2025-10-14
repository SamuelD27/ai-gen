"""Add error_message column to training_sessions table

This migration adds the error_message column to store detailed error information
when training sessions fail, enabling better debugging and user feedback.
"""

from sqlalchemy import text
from app.core.database import engine

def upgrade():
    """Add error_message column to training_sessions table."""
    with engine.connect() as conn:
        # First check if the table exists
        table_check = conn.execute(text("""
            SELECT COUNT(*) as count
            FROM sqlite_master
            WHERE type='table' AND name='training_sessions'
        """))

        table_exists = table_check.fetchone()[0] > 0

        if not table_exists:
            print("⚠ training_sessions table doesn't exist yet - creating all tables first")
            # Import Base and all models to populate metadata
            from app.core.database import (
                Base, User, Character, TrainingSession, InferenceJob,
                AppSettings, Dataset, DatasetImage
            )
            Base.metadata.create_all(engine)
            print("✓ Database tables created")

        # Now check if column exists
        result = conn.execute(text("""
            SELECT COUNT(*) as count
            FROM pragma_table_info('training_sessions')
            WHERE name = 'error_message'
        """))

        exists = result.fetchone()[0] > 0

        if not exists:
            # Add error_message column
            conn.execute(text("""
                ALTER TABLE training_sessions
                ADD COLUMN error_message TEXT
            """))
            conn.commit()
            print("✓ Added error_message column to training_sessions table")
        else:
            print("✓ error_message column already exists, skipping migration")

def downgrade():
    """Remove error_message column from training_sessions table."""
    # Note: SQLite doesn't support DROP COLUMN directly
    # Would require recreating the table, but for safety we'll just warn
    print("⚠ Warning: SQLite doesn't support DROP COLUMN directly")
    print("⚠ To remove error_message column, you would need to recreate the table")
    print("⚠ Skipping downgrade for safety")

if __name__ == "__main__":
    upgrade()
