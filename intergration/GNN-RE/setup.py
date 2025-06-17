import os
import shutil
import subprocess
import sys





def main():
    original_dir = os.getcwd()
    directory = './Netlist_to_graph/Circuits_datasets/Interconnected-Modules'
    save_path = './Netlist_to_graph/Circuits_datasets/Interconnected-Modules/SAVE'
    graphs_directory_path = './Netlist_to_graph/Graphs_datasets/'


    parsers_path = '../../Parsers/netlist_to_graph_re.pl'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    entries = os.listdir(directory)

    test_files = [f for f in entries  if f.startswith("Test")]
    
    source_folder = directory
    target_folder = save_path
    for filename in test_files:
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(target_folder, filename)
        shutil.move(src_path, dst_path)
        
        
    circuits_path_perl = os.path.join('../..', 'Circuits_datasets', 'Interconnected-Modules')
    for filename in test_files:
        src_path = os.path.join(target_folder, filename)
        dst_path = os.path.join(source_folder, filename)
        shutil.move(src_path, dst_path)
        
        verilog_name = filename.replace(".v", "")
        print( "Now processing: ", verilog_name)
        graphs_path = os.path.join(graphs_directory_path, verilog_name)
        
        if not os.path.exists(graphs_path):
            os.makedirs(graphs_path)
        shutil.copy('graph_parser_cirstag.py', graphs_path)
        shutil.copy('graph_parser_directed_full_cirstag.py', graphs_path)
        shutil.copy('netlist_to_graph_re_cirstag.pl', graphs_path)
        shutil.copy('theCircuit_cirstag.pm', graphs_path)
        
        os.chdir(graphs_path)
        
        try:
            perl_command = ['perl', 'netlist_to_graph_re_cirstag.pl', '-i', circuits_path_perl]
            subprocess.run(perl_command, check=True)
            print(f"Perl parser executed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to run Perl script in {graphs_path}: {e}")
            sys.exit(1)
        
        try:
            subprocess.run(['python3', 'graph_parser_cirstag.py'], check=True)
            print(f"Py parser executed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to run Python script in {graphs_path}: {e}")
            sys.exit(1)
            
        try:
            subprocess.run(['python3', 'graph_parser_directed_full_cirstag.py'], check=True)
            print(f"Py parser directed executed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to run Python script in {graphs_path}: {e}")
            sys.exit(1)
        
        os.chdir(original_dir)
        
        shutil.move(dst_path, src_path)
        
    for filename in test_files:
        src_path = os.path.join(target_folder, filename)
        dst_path = os.path.join(source_folder, filename)
        shutil.move(src_path, dst_path)
    
    
    



if __name__ == "__main__":
    main()
