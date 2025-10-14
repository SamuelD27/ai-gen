#!/usr/bin/env python3
"""
Database Migration Runner

Runs all pending migrations in the migrations/ directory.
Migrations are run in alphabetical order by filename.

Usage:
    python run_migrations.py              # Run all migrations
    python run_migrations.py --list       # List available migrations
    python run_migrations.py --single <filename>  # Run specific migration
"""

import sys
import os
from pathlib import Path
import importlib.util

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def load_migration(migration_file: Path):
    """Load a migration module dynamically."""
    spec = importlib.util.spec_from_file_location(
        migration_file.stem,
        migration_file
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def list_migrations():
    """List all available migrations."""
    migrations_dir = backend_dir / "migrations"
    migration_files = sorted(migrations_dir.glob("*.py"))
    migration_files = [f for f in migration_files if f.stem != "__init__"]

    print("\n📋 Available Migrations:")
    print("=" * 60)
    for migration_file in migration_files:
        print(f"  • {migration_file.name}")
    print("=" * 60)
    print(f"\nTotal: {len(migration_files)} migrations\n")

def run_migrations(single_migration: str = None):
    """Run all migrations or a single migration."""
    migrations_dir = backend_dir / "migrations"

    if not migrations_dir.exists():
        print("❌ Migrations directory not found!")
        return False

    # Get list of migration files
    if single_migration:
        migration_files = [migrations_dir / single_migration]
        if not migration_files[0].exists():
            print(f"❌ Migration file not found: {single_migration}")
            return False
    else:
        migration_files = sorted(migrations_dir.glob("*.py"))
        migration_files = [f for f in migration_files if f.stem != "__init__"]

    if not migration_files:
        print("✓ No migrations to run")
        return True

    print("\n🚀 Running Database Migrations")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for migration_file in migration_files:
        print(f"\n📦 Running: {migration_file.name}")
        print("-" * 60)

        try:
            # Load and run migration
            migration = load_migration(migration_file)

            if not hasattr(migration, 'upgrade'):
                print(f"⚠ Warning: {migration_file.name} has no upgrade() function, skipping")
                continue

            migration.upgrade()
            success_count += 1

        except Exception as e:
            print(f"❌ Error running {migration_file.name}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    print("\n" + "=" * 60)
    print(f"✓ Completed: {success_count} successful, {error_count} errors")
    print("=" * 60 + "\n")

    return error_count == 0

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run database migrations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_migrations.py                    # Run all migrations
  python run_migrations.py --list             # List available migrations
  python run_migrations.py --single add_error_message_column.py  # Run specific migration
        """
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available migrations'
    )

    parser.add_argument(
        '--single',
        type=str,
        metavar='FILENAME',
        help='Run a single migration file'
    )

    args = parser.parse_args()

    if args.list:
        list_migrations()
        sys.exit(0)

    success = run_migrations(single_migration=args.single)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
