#!/bin/bash

##### These lines are for SLURM
#SBATCH -N 37
#SBATCH -p pbatch
#SBATCH -A paratime
#SBATCH -t 4
#SBATCH -o out.%j
#SBATCH -e err.%j
##### These are shell commands

echo -n 'This machine is '; hostname
echo -n 'My jobid is '; echo $SLURM_JOBID
echo -n 'Timestamp START: ';date

# number of cores
ncores="2048"
# ncores="1 2 4 8"

# levels
mlevels="4"

# fixed arguments
fargs="-tf 4 -nt 16384 -cf0 16 -cf 4 -theta -nu 1 -niters 1"

# path to executable
ex="../../drive-lorenz-Delta"

# output directory
outd="."

# output fname
outn="lorenz_theta"

for nc in $ncores; do
   for ml in $mlevels; do
      # echo "srun -N 2 -o ${outd}/${outn}_Delta_nc${nc}_ml${ml} -n ${nc} ${ex} ${fargs} -ml ${ml} -Delta"
      srun -N 19 -n ${nc} -o ${outd}/${outn}_Delta_fmg_nc${nc}_ml${ml} ${ex} ${fargs} -ml ${ml} -Delta -fmg
      srun -N 19 -n ${nc} -o ${outd}/${outn}_Delta_nc${nc}_ml${ml}     ${ex} ${fargs} -ml ${ml} -Delta
      srun -N 19 -n ${nc} -o ${outd}/${outn}_nc${nc}_ml${ml}           ${ex} ${fargs} -ml ${ml}
      # mpirun -n ${nc} ${ex} ${fargs} -ml ${ml} -Delta > ${outd}/${outn}_Delta_nc${nc}_ml${ml}
      # mpirun -n ${nc} ${ex} ${fargs} -ml ${ml}        > ${outd}/${outn}_nc${nc}_ml${ml}
   done
done

echo 'Done'
echo -n 'Timestamp END: ';date
