import argparse
import os

batchfile="""#!/bin/bash
#SBATCH --job-name={name}.job
#SBATCH --time=168:00:00
#SBATCH --partition=mhg
#SBATCH --account=mhg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={nt}
#SBATCH --array={arr_range}%40
#SBATCH --output={name}_%A_task_%a.job.o
#SBATCH --error={name}_%A_task_%a.job.err

source $HOME/.bashrc
cd {job_directory} 
qchem -nt {nt} samp_$SLURM_ARRAY_TASK_ID.qcin ../output_file/samp_$SLURM_ARRAY_TASK_ID.q.out samp_$SLURM_ARRAY_JOBID.$SLURM_ARRAY_TASK_ID.o$SLURM_JOBID
"""

def main():
    path = os.path.abspath(os.getcwd())
    print(path)
    create_output_directory(path)
    jobfile = write_jobfile(*(retrieve_specs(path)))
    run(jobfile)

def retrieve_specs(path_to_directory):
    """
    File of the form:
    JOBNAME
    -array 1-100
    -nt 6
    in the 'files_to_run' directory
    """
    job_directory = os.getcwd()
    filename = os.path.join(job_directory,'specs')
    with open(filename, 'r') as f:
        name = f.readline().split()[0]
        arr_range = f.readline().split()[-1]
        nt = f.readline().split()[-1]
        f.close()
    print(job_directory)
    return job_directory, name, arr_range, nt 

def create_output_directory(path_to_directory):
    if not os.path.exists(path_to_directory):
        os.makedirs(path_to_directory)
    output_path = os.path.join(path_to_directory,'output_file')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

def write_jobfile(job_directory, name, arr_range, nt):
    jobfile = os.path.join(job_directory,'{}.job'.format(name))
    with open(jobfile, 'w') as f:
        f.write(batchfile.format(name=name,nt=nt,arr_range=arr_range,job_directory=job_directory))
        f.close() 
    return jobfile

def get_project_directory(command_line_args=None):
    parser = argparse.ArgumentParser(description='Create and run array jobs for sampling')
    parser.add_argument('-dir', type=str, nargs=1, 
        help='Project directory for sampling. Default: current working directory.')
    args = parser.parse_args(command_line_args)
    try:
        directory = args.dir[0]
    except TypeError:
        directory = os.getcwd()
    return os.path.abspath(directory)

def run(jobfile):
    os.system('sbatch {}'.format(jobfile))

if __name__=='__main__':
    main()
