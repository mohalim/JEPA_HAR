def save_history_txt(history, file_path):
    """
    Save training history to a human-readable TXT file.

    Args:
        history (list of dict): List containing training/validation metrics per epoch.
        file_path (str): Path to save the TXT file.
    """
    with open(file_path, "a") as f:
        # Write header
        headers = history[0].keys()
        f.write(" | ".join(headers) + "\n")
        
        # Write each record
        for record in history:
            line = " | ".join(
                f"{v:.4f}" if isinstance(v, float) else str(v) 
                for v in record.values()
            )
            f.write(line + "\n")
    
    print(f"Training history saved to {file_path}")
