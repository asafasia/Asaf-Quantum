import os


def find_main_project_directory():
    current_directory = os.getcwd()
    marker_name = 'marker_file.txt'
    while True:
        # Check if the marker file or folder exists in the current directory
        marker_path = os.path.join(current_directory, marker_name)

        if os.path.exists(marker_path):
            return current_directory  # Found the main project directory

        # Move up one level in the directory hierarchy
        parent_directory = os.path.dirname(current_directory)

        # Check if we've reached the root of the file system
        if parent_directory == current_directory:
            raise Exception(f"Marker '{marker_name}' not found in the directory tree.")

        current_directory = parent_directory


project_path = find_main_project_directory()
