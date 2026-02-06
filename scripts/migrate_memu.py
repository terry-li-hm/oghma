#!/usr/bin/env python3
"""
Migrate memU memories to Oghma SQLite database.
Idempotent script with dry-run support.
"""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple


# Category mapping from memU to Oghma
CATEGORY_MAPPING = {
    'personal_info': 'project_context',
    'relationships': 'project_context',
    'preferences': 'preference',
    'opinions': 'preference',
    'habits': 'preference',
    'knowledge': 'learning',
    'learnings': 'learning',
    'work': 'workflow',
    'work_life': 'workflow',
    'experiences': 'workflow',
    'activities': 'workflow',
    'goals': 'project_context',
}


def parse_memu_file(file_path: Path) -> List[Tuple[str, str]]:
    """Parse a memU markdown file and extract items.
    
    Returns:
        List of (content, type) tuples.
    """
    items = []
    content = file_path.read_text(encoding='utf-8')
    
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines, headers, and "--- Updated" footer
        if not line:
            continue
        if line.startswith('#'):
            continue
        if line.startswith('---'):
            continue
        
        # Parse bullet points: - **type**: content
        if line.startswith('- **'):
            try:
                # Extract type and content
                type_part, content_part = line.split('**: ', 1)
                memory_type = type_part.replace('- **', '').strip()
                content_text = content_part.strip()
                
                if content_text:
                    items.append((content_text, memory_type))
            except (ValueError, IndexError):
                # Skip malformed lines
                continue
    
    return items


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content for deduplication."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def get_category_from_filename(filename: str) -> str:
    """Map memU filename to Oghma category."""
    base_name = filename.replace('.md', '')
    return CATEGORY_MAPPING.get(base_name, 'workflow')


def parse_memu_directory(source_dir: Path) -> Dict[str, List[Dict]]:
    """Parse all memU files and return structured data.
    
    Returns:
        Dict mapping category to list of items with metadata.
    """
    memu_files = [
        'activities.md', 'experiences.md', 'goals.md', 'habits.md',
        'knowledge.md', 'learnings.md', 'opinions.md', 'personal_info.md',
        'preferences.md', 'relationships.md', 'work.md', 'work_life.md',
    ]
    
    categorized_items: Dict[str, List[Dict]] = {}
    
    for filename in memu_files:
        file_path = source_dir / filename
        if not file_path.exists():
            print(f"Warning: {filename} not found, skipping")
            continue
        
        items = parse_memu_file(file_path)
        category = get_category_from_filename(filename)
        
        if category not in categorized_items:
            categorized_items[category] = []
        
        for content, memory_type in items:
            categorized_items[category].append({
                'content': content,
                'memory_type': memory_type,
                'source_file': str(file_path),
                'content_hash': compute_content_hash(content),
            })
    
    return categorized_items


def dry_run(db_path: Path, categorized_items: Dict[str, List[Dict]]) -> Dict[str, int]:
    """Simulate import and return statistics."""
    stats = {
        'items_found': 0,
        'duplicates': 0,
        'to_import': 0,
        'by_category': {},
    }
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for category, items in categorized_items.items():
        category_stats = {'found': 0, 'duplicates': 0, 'to_import': 0}
        
        for item in items:
            category_stats['found'] += 1
            stats['items_found'] += 1
            
            # Check if already exists
            cursor.execute(
                "SELECT id FROM memories WHERE content_hash = ? AND source_file = ?",
                (item['content_hash'], item['source_file'])
            )
            
            if cursor.fetchone():
                category_stats['duplicates'] += 1
                stats['duplicates'] += 1
            else:
                category_stats['to_import'] += 1
                stats['to_import'] += 1
        
        stats['by_category'][category] = category_stats
    
    conn.close()
    return stats


def import_to_oghma(db_path: Path, categorized_items: Dict[str, List[Dict]]) -> Dict[str, int]:
    """Import memories to Oghma database.
    
    Returns:
        Dictionary with import statistics.
    """
    stats = {
        'items_found': 0,
        'duplicates': 0,
        'imported': 0,
        'by_category': {},
    }
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Single transaction for performance
        with conn:
            for category, items in categorized_items.items():
                category_stats = {'found': 0, 'duplicates': 0, 'imported': 0}
                
                for item in items:
                    category_stats['found'] += 1
                    stats['items_found'] += 1
                    
                    # Check for duplicates
                    cursor.execute(
                        "SELECT id FROM memories WHERE content_hash = ? AND source_file = ?",
                        (item['content_hash'], item['source_file'])
                    )
                    
                    if cursor.fetchone():
                        category_stats['duplicates'] += 1
                        stats['duplicates'] += 1
                        continue
                    
                    # Build metadata JSON
                    metadata = {
                        'memory_type': item['memory_type'],
                        'original_category': category,
                    }
                    
                    # Insert new memory
                    cursor.execute(
                        """INSERT INTO memories (
                            content, category, source_tool, source_file,
                            confidence, status, metadata, content_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            item['content'],
                            category,
                            'memu_import',
                            item['source_file'],
                            1.0,
                            'active',
                            json.dumps(metadata),
                            item['content_hash'],
                        )
                    )
                    
                    category_stats['imported'] += 1
                    stats['imported'] += 1
                
                stats['by_category'][category] = category_stats
    
    finally:
        conn.close()
    
    return stats


def print_summary(stats: Dict[str, int], dry_run: bool = False):
    """Print import summary."""
    action = "Would import" if dry_run else "Imported"
    
    print(f"\n{'='*60}")
    print(f"memU Migration Summary ({'DRY RUN' if dry_run else 'LIVE'})")
    print(f"{'='*60}\n")
    
    print(f"Total items found:      {stats['items_found']}")
    print(f"Duplicates skipped:     {stats['duplicates']}")
    print(f"New items {action.lower()}: {stats['imported' if not dry_run else 'to_import']}")
    
    print(f"\n{'='*60}")
    print("By Category:")
    print(f"{'='*60}\n")
    
    for category, cat_stats in stats['by_category'].items():
        action_word = "Import" if not dry_run else "Would import"
        print(f"{category:20} | Found: {cat_stats['found']:4} | "
              f"Skip: {cat_stats['duplicates']:3} | {action_word}: {cat_stats['imported' if not dry_run else 'to_import']:3}")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate memU memories to Oghma SQLite database'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be imported without actually doing it'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='/Users/terry/notes/memory/memu/',
        help='Source directory containing memU markdown files'
    )
    parser.add_argument(
        '--database',
        type=str,
        default='/Users/terry/.oghma/oghma.db',
        help='Path to Oghma SQLite database'
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    db_path = Path(args.database)
    
    # Validate paths
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return 1
    
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        return 1
    
    # Parse memU files
    print(f"Parsing memU files from: {source_dir}")
    categorized_items = parse_memu_directory(source_dir)
    
    if not categorized_items:
        print("No memU files found or no items to import.")
        return 0
    
    # Run dry-run or live import
    if args.dry_run:
        print(f"\nRunning dry-run against database: {db_path}")
        stats = dry_run(db_path, categorized_items)
        print_summary(stats, dry_run=True)
    else:
        print(f"\nImporting to database: {db_path}")
        stats = import_to_oghma(db_path, categorized_items)
        print_summary(stats, dry_run=False)
    
    return 0


if __name__ == '__main__':
    exit(main())
