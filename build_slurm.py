draft_head_cascade = """#!/bin/bash
#SBATCH --job-name="{0}"
#SBATCH --partition=gpu
#SBATCH --constraint="cascade"
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=4
#SBATCH --account=chenGrp
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/{0}-%j.o"
#SBATCH --error="jobs_oe/{0}-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
"""

draft_head_cpu = """#!/bin/bash
#SBATCH --job-name="{0}"
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=50GB
#SBATCH --ntasks-per-node=8
#SBATCH --account=chenGrp
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/{0}-%j.o"
#SBATCH --error="jobs_oe/{0}-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
"""

draft_normal = "python {0} {1}\n"


def one_slurm_multi_seed(job_name, python_name, kwargs, draft_head=draft_head_cpu, draft_normal=draft_normal, device="cpu"):
    full_path = "jobs/{}.slurm".format(job_name)
    assert device in ["skylake", "cascade", "cpu"]
    draft_used = draft_head
    if device == "cpu":
        draft_used = draft_head_cpu
    if device == "cascade":
        draft_used = draft_head_cascade
    print("build {}".format(full_path))
    with open(full_path, "w") as f:
        f.write(draft_used.format(
            job_name
        ))

        f.write(draft_normal.format(
            python_name,
            " ".join(["--{0} {1}".format(one_key, kwargs[one_key]) for one_key in kwargs])
        ))


def one_time_build_slurm():
    # --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.00001 --beta 1 --h_dim 8
    # num / alpha / h_dim / keep_prob
    plans = [
        [1, 0.0001, 7, 0.6],
        [1, 0.0001, 8, 0.6],
        [1, 0.0001, 9, 0.6],
        [1, 0.00001, 7, 0.6],
        [1, 0.00001, 8, 0.6],
        [1, 0.00001, 9, 0.6],
        [1, 0.000001, 7, 0.6],
        [1, 0.000001, 8, 0.6],
        [1, 0.000001, 9, 0.6],

        [1, 0.0001, 7, 0.7],
        [1, 0.0001, 8, 0.7],
        [1, 0.0001, 9, 0.7],
        [1, 0.00001, 7, 0.7],
        [1, 0.00001, 8, 0.7],
        [1, 0.00001, 9, 0.7],
        [1, 0.000001, 7, 0.7],
        [1, 0.000001, 8, 0.7],
        [1, 0.000001, 9, 0.7],

        [1, 0.0001, 7, 0.8],
        [1, 0.0001, 8, 0.8],
        [1, 0.0001, 9, 0.8],
        [1, 0.00001, 7, 0.8],
        [1, 0.00001, 8, 0.8],
        [1, 0.00001, 9, 0.8],
        [1, 0.000001, 7, 0.8],
        [1, 0.000001, 8, 0.8],
        [1, 0.000001, 9, 0.8],

        [1, 0.0001, 7, 0.9],
        [1, 0.0001, 8, 0.9],
        [1, 0.0001, 9, 0.9],
        [1, 0.00001, 7, 0.9],
        [1, 0.00001, 8, 0.9],
        [1, 0.00001, 9, 0.9],
        [1, 0.000001, 7, 0.9],
        [1, 0.000001, 8, 0.9],
        [1, 0.000001, 9, 0.9],
    ]
    for one_plan in plans:
        dic = dict()
        dic["num"] = one_plan[0]
        dic["alpha"] = one_plan[1]
        dic["h_dim"] = one_plan[2]
        dic["keep_prob"] = one_plan[3]
        dic["data"] = "eta1"
        dic["k"] = 6
        dic["beta"] = 1.0
        dic["kmeans"] = 1
        dic["main_epoch"] = 1000

        title_format = "{}_alpha={}_h_dim={}_keep_prob={}".format(dic["data"], dic["alpha"], dic["h_dim"], dic["keep_prob"])

        one_slurm_multi_seed(
            title_format,
            "run.py", dic,
            device="cpu",
        )


if __name__ == "__main__":
    one_time_build_slurm()
