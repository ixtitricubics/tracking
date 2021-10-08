if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
	file_path="/home/ixti/Documents/projects/video/wallet_cup.mkv"
  else
	file_path=$1
fi

echo "opening file: $file_path"

python darknet_video.py --input $file_path --weights yolov4_final.weights --config_file yolov4.cfg --data_file drink_ix2.data --thresh 0.9
