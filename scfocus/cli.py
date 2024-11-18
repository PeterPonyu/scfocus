import argparse  
import sys  
import os  
import subprocess  

def run_streamlit():  
    """运行 Streamlit 应用"""  
    # 获取当前包的安装路径  
    import scfocus  
    package_dir = os.path.dirname(os.path.abspath(scfocus.__file__))  
    streamlit_app_path = os.path.join(package_dir, 'Analysis.py')  
    
    try:  
        # 使用 subprocess 运行 streamlit  
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
    
    # 添加子命令  
    subparsers = parser.add_subparsers(dest='command', help='Available commands')  
    
    # 处理数据的子命令  
    process_parser = subparsers.add_parser('process', help='Process single cell data')  
    process_parser.add_argument('--input', '-i', type=str, required=True,  
                              help='Input file path (h5ad format)')  
    process_parser.add_argument('--output', '-o', type=str, default='output.h5ad',  
                              help='Output file path (default: output.h5ad)')  
    
    # 可视化的子命令  
    visualize_parser = subparsers.add_parser('visualize', help='Visualize results')  
    visualize_parser.add_argument('--input', '-i', type=str, required=True,  
                                help='Input file path (processed h5ad file)')  
    
    # 添加 streamlit 子命令  
    streamlit_parser = subparsers.add_parser('ui',   
                                           help='Launch the Streamlit web interface')  
    
    args = parser.parse_args()  
    
    if args.command == 'process':  
        # 处理数据  
        # from scfocus import process_data  
        # process_data(args.input, args.output)  
        pass
    elif args.command == 'visualize':  
        # 可视化结果  
        # from scfocus import visualize_results  
        # visualize_results(args.input) 
        pass 
    elif args.command == 'ui':  
        # 运行 Streamlit 应用  
        run_streamlit()  
    else:  
        parser.print_help()  

if __name__ == '__main__':  
    main()
