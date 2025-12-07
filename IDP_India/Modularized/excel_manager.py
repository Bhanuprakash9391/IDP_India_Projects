import pandas as pd
import os
from datetime import datetime
from typing import Dict, List
import json

def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    """
    Flatten a nested dictionary
    Example: {'key': {'nested': 'value'}} -> {'key_nested': 'value'}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to string representation
            items.append((new_key, json.dumps(v)))
        else:
            items.append((new_key, v))
    return dict(items)

def save_structured_data_to_excel(per_file_data: Dict, output_dir: str = "exported_data"):
    """
    Group documents by type and save structured data (flattened) to separate Excel files.

    Only includes:
      - File name
      - Document type
      - Classification confidence
      - Flattened structured data extracted from the document

    Args:
        per_file_data: Dictionary with file names as keys and their extracted data.
        output_dir: Directory to save Excel files.

    Returns:
        Dictionary with document types and their saved file paths.
    """
    import os
    import json
    import pandas as pd
    from datetime import datetime

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Helper: Convert 0-based index to Excel column name (A, B, ..., Z, AA, AB, etc.)
    def get_excel_column_name(n: int) -> str:
        name = ''
        while n >= 0:
            n, remainder = divmod(n, 26)
            name = chr(65 + remainder) + name
            n -= 1
        return name

    # Group files by document type
    grouped_data = {}

    for file_name, file_data in per_file_data.items():
        classification = file_data.get('classification', {})
        doc_type = classification.get('document_type', 'Unknown').replace('/', '_').replace('\\', '_')

        if doc_type not in grouped_data:
            grouped_data[doc_type] = []

        # Prepare row data
        row_data = {
            'File_Name': file_name,
            'Document_Type': doc_type,
            'Classification_Confidence': classification.get('confidence', 0)
        }

        # Add structured data (flattened)
        structured_data = file_data.get('structured_data', {})
        if structured_data and 'error' not in structured_data:
            flattened_structured = flatten_dict(structured_data)
            row_data.update(flattened_structured)

        grouped_data[doc_type].append(row_data)

    # Save each document type to separate Excel file
    saved_files = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for doc_type, records in grouped_data.items():
        if not records:
            continue

        # Create DataFrame
        df = pd.DataFrame(records)

        # Generate filename
        safe_doc_type = doc_type.replace(' ', '_').replace('/', '_')
        filename = f"{safe_doc_type}_{timestamp}.xlsx"
        filepath = os.path.join(output_dir, filename)

        # Save to Excel with formatting
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            worksheet = writer.sheets['Data']

            # Auto-adjust column widths (handles >26 columns)
            for idx, col in enumerate(df.columns):
                col_letter = get_excel_column_name(idx)
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(str(col))
                )
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[col_letter].width = adjusted_width

        saved_files[doc_type] = {
            'filepath': filepath,
            'count': len(records),
            'filename': filename
        }

        print(f"✓ Saved {len(records)} {doc_type} structured document(s) to {filename}")

    # Create summary Excel file
    summary_data = [
        {'Document_Type': doc_type, 'Count': info['count'], 'Excel_File': info['filename']}
        for doc_type, info in saved_files.items()
    ]

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_filepath = os.path.join(output_dir, f"Summary_{timestamp}.xlsx")
        summary_df.to_excel(summary_filepath, index=False)
        print(f"\n✓ Summary saved to {summary_filepath}")

    return saved_files



