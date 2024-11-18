import argparse  
import sys  
import os  
import subprocess  

def run_streamlit():  
    """Run the Streamlit application"""  
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
    process_parser = subparsers.add_parser('process', help='Process single cell data')  
    process_parser.add_argument('--input', '-i', type=str, required=True,  
                              help='Input file path (h5ad format)')  
    process_parser.add_argument('--output', '-o', type=str, default='output.h5ad',  
                              help='Output file path (default: output.h5ad)')  
    
    # Subcommand for visualization  
    visualize_parser = subparsers.add_parser('visualize', help='Visualize results')  
    visualize_parser.add_argument('--input', '-i', type=str, required=True,  
                                help='Input file path (processed h5ad file)')  
    
    # Add streamlit subcommand  
    streamlit_parser = subparsers.add_parser('ui',   
                                           help='Launch the Streamlit web interface')  
    
    args = parser.parse_args()  
    
    if args.command == 'process':  
        raise NotImplementedError("The 'process' command is not implemented yet.")  
    elif args.command == 'visualize':  
        raise NotImplementedError("The 'visualize' command is not implemented yet.")  
    elif args.command == 'ui':  
        # Run the Streamlit application  
        run_streamlit()  
    else:  
        parser.print_help()  

if __name__ == '__main__':  
    main()

