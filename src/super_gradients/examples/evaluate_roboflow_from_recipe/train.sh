# Example: sh train.sh roboflow_ppyoloe results.csv
# Note: tweeter-profile seems to be lacking data so it was removed

if [ -z "$1" ]
then
  echo "Please provide a config name as an argument"
  exit 1
fi

if [ -z "$2" ]
then
  echo "Please provide a output file name as an argument"
  exit 1
fi


counter=0
configname="$1"
outputfile="$2"
datasets=$(pwd)/datasets.txt

outputfile_fullpath=$(pwd)/$outputfile
echo "\nResults will be saved in ${outputfile_fullpath}"

awk -F "," '{print $1}' $datasets | while read dataset_name; do
    echo "\n\n\n> [${counter}/100] ${dataset_name}\n\n"
    python -u ../train_from_recipe_example/train_from_recipe.py --config-name=$configname dataset_name=$dataset_name result_path=$outputfile_fullpath
    counter=$((counter + 1))
    echo "\n> ${dataset_name} training over... \n"
done

echo "Results saved in ${outputfile_fullpath}"
echo ">> DONE"
