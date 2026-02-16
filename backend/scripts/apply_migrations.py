#!/usr/bin/env python3
"""
Database Migration Application Script
Applies SQL migrations to Supabase database

Usage:
    python scripts/apply_migrations.py [--migration MIGRATION_FILE]
    
Examples:
    python scripts/apply_migrations.py  # Apply all migrations
    python scripts/apply_migrations.py --migration 002_create_body_generations_table.sql
"""

import os
import sys
from pathlib import Path
from supabase import create_client, Client
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Migration directory
MIGRATIONS_DIR = Path(__file__).parent.parent / "database_migrations"


def get_supabase_client() -> Client:
    """Create and return Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        sys.exit(1)
    
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def read_migration_file(filepath: Path) -> str:
    """Read SQL migration file content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ Error: Migration file not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading migration file: {e}")
        sys.exit(1)


def apply_migration(client: Client, migration_name: str, sql_content: str) -> bool:
    """
    Apply a single migration to the database.
    
    Note: Supabase Python client doesn't support direct SQL execution.
    This script provides the SQL content for manual execution in Supabase SQL Editor.
    """
    print(f"\n{'='*60}")
    print(f"Migration: {migration_name}")
    print(f"{'='*60}")
    print("\n⚠️  IMPORTANT: Supabase Python client doesn't support direct SQL execution.")
    print("Please copy the SQL below and execute it in your Supabase SQL Editor:")
    print(f"\n{sql_content}\n")
    print(f"{'='*60}\n")
    
    response = input("Have you executed this migration in Supabase SQL Editor? (yes/no): ")
    return response.lower() in ['yes', 'y']


def get_migration_files() -> list[Path]:
    """Get all migration files sorted by version number."""
    if not MIGRATIONS_DIR.exists():
        print(f"❌ Error: Migrations directory not found: {MIGRATIONS_DIR}")
        sys.exit(1)
    
    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    
    if not migration_files:
        print(f"⚠️  Warning: No migration files found in {MIGRATIONS_DIR}")
        return []
    
    return migration_files


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Apply database migrations to Supabase")
    parser.add_argument(
        "--migration",
        type=str,
        help="Specific migration file to apply (e.g., 002_create_body_generations_table.sql)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available migrations"
    )
    
    args = parser.parse_args()
    
    # Get Supabase client
    client = get_supabase_client()
    print(f"✅ Connected to Supabase: {SUPABASE_URL}")
    
    # Get migration files
    migration_files = get_migration_files()
    
    if args.list:
        print("\n📋 Available migrations:")
        for i, filepath in enumerate(migration_files, 1):
            print(f"  {i}. {filepath.name}")
        return
    
    # Filter to specific migration if requested
    if args.migration:
        migration_path = MIGRATIONS_DIR / args.migration
        if not migration_path.exists():
            print(f"❌ Error: Migration file not found: {migration_path}")
            sys.exit(1)
        migration_files = [migration_path]
    
    # Apply migrations
    print(f"\n🚀 Applying {len(migration_files)} migration(s)...\n")
    
    applied_count = 0
    for filepath in migration_files:
        sql_content = read_migration_file(filepath)
        
        if apply_migration(client, filepath.name, sql_content):
            applied_count += 1
            print(f"✅ Migration {filepath.name} marked as applied")
        else:
            print(f"⏭️  Skipping migration {filepath.name}")
    
    print(f"\n{'='*60}")
    print(f"✅ Migration process complete!")
    print(f"   Applied: {applied_count}/{len(migration_files)} migrations")
    print(f"{'='*60}\n")
    
    if applied_count < len(migration_files):
        print("⚠️  Some migrations were skipped. Run again to apply remaining migrations.")


if __name__ == "__main__":
    main()
