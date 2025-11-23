import argparse  
import sys  
import os  
import subprocess  

def run_streamlit():  
    """  
    Launch the Streamlit web application for scFocus.  
    
    This function locates the Analysis.py file in the scfocus package directory  
    and launches it using the streamlit command-line interface.  
    
    Raises  
    ------  
    subprocess.CalledProcessError  
        If the Streamlit application fails to launch.  
    FileNotFoundError  
        If Streamlit is not installed in the current environment.  
    
    Notes  
    -----  
    This function is called when the user runs `scfocus ui` from the command line.  
    The Streamlit application provides an interactive web interface for analyzing  
    single-cell data without writing code.  
    """
    # Get the installation path of the current package  
    import scfocus  
    package_dir = os.path.dirname(os.path.abspath(scfocus.__file__))  
    streamlit_app_path = os.path.join(package_dir, 'Analysis.py')  
    
    try:  
        # Run streamlit using subprocess  
        subprocess.run(['streamlit', 'run', streamlit_app_path], check=True)  
    except subprocess.CalledProcessError as e:  
        print(f"Error running Streamlit app: {str(e)}", file=sys.stderr)  
        sys.exit(1)  
    except FileNotFoundError:  
        print("Error: Streamlit is not installed. Please install it using 'pip install streamlit'",   
              file=sys.stderr)  
        sys.exit(1)  

def main():  
    """  
    Main entry point for the scFocus command-line interface.  
    
    This function parses command-line arguments and dispatches to the appropriate  
    subcommand handler. Available commands include:  
    
    - ui: Launch the Streamlit web interface  
    - process: Process single-cell data (planned for future release)  
    - visualize: Visualize analysis results (planned for future release)  
    
    The function displays help information if no command is specified or if  
    an invalid command is provided.  
    """
    parser = argparse.ArgumentParser(  
        description='''  
        scFocus: Single Cell Reinforcement Learning for Lineage Focusing  
        
        This tool processes single-cell data using reinforcement learning  
        techniques to focus on relevant features and patterns on cell lineage.  
        ''',  
        formatter_class=argparse.RawDescriptionHelpFormatter  
    )  
    
    # Add subcommands  
    subparsers = parser.add_subparsers(dest='command', help='Available commands')  
    
    # Subcommand for data processing  
    process_parser = subparsers.add_parser('process', help='Process single cell data (coming soon)')  
    process_parser.add_argument('--input', '-i', type=str, required=True,  
                              help='Input file path (h5ad format)')  
    process_parser.add_argument('--output', '-o', type=str, default='output.h5ad',  
                              help='Output file path (default: output.h5ad)')  
    
    # Subcommand for visualization  
    visualize_parser = subparsers.add_parser('visualize', help='Visualize results (coming soon)')  
    visualize_parser.add_argument('--input', '-i', type=str, required=True,  
                                help='Input file path (processed h5ad file)')  
    
    # Add streamlit subcommand  
    streamlit_parser = subparsers.add_parser('ui',   
                                           help='Launch the Streamlit web interface')  
    
    args = parser.parse_args()  
    
    if args.command == 'process':  
        raise NotImplementedError(
            "The 'process' command is not implemented yet. "
            "Please use the web interface (scfocus ui) or the Python API for now."
        )
    elif args.command == 'visualize':  
        raise NotImplementedError(
            "The 'visualize' command is not implemented yet. "
            "Please use the web interface (scfocus ui) or the Python API for now."
        )
    elif args.command == 'ui':  
        # Run the Streamlit application  
        run_streamlit()  
    else:  
        parser.print_help()  

if __name__ == '__main__':  
    main()

