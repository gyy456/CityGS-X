# MatrixCity, Aerial View, block All
mkdir datasets/MatrixCity/aerial/train/block_all
mkdir datasets/MatrixCity/aerial/test/block_all_test
mkdir datasets/MatrixCity/aerial/train/block_all/input
mkdir datasets/MatrixCity/aerial/test/block_all_test/input
cp datasets/MatrixCity/aerial/pose/block_all/transforms_train.json datasets/MatrixCity/aerial/train/block_all/transforms.json
cp datasets/MatrixCity/aerial/pose/block_all/transforms_test.json datasets/MatrixCity/aerial/test/block_all_test/transforms.json

# Gather images and initialize sparse folder
python tools/transform_json2txt_mc_aerial.py --source_path datasets/MatrixCity/aerial/train/block_all
# python tools/transform_json2txt_mc_aerial.py --source_path datasets/MatrixCity/aerial/test/block_all_test

# Remove the old sparse folder and use the downloaded one
rm -rf datasets/MatrixCity/aerial/train/block_all/sparse
rm -rf datasets/MatrixCity/aerial/test/block_all_test/sparse

# mv data/colmap_results/matrix_city_aerial/train/sparse datasets/MatrixCity/aerial/train/block_all
# mv data/colmap_results/matrix_city_aerial/test/sparse datasets/MatrixCity/aerial/test/block_all_test

# # MatrixCity, Street View, block A
# mkdir datasets/MatrixCity/street/train/block_A
# mkdir datasets/MatrixCity/street/test/block_A_test
# mkdir datasets/MatrixCity/street/train/block_A/input
# mkdir datasets/MatrixCity/street/test/block_A_test/input
# cp datasets/MatrixCity/street/pose/block_A/transforms_train.json datasets/MatrixCity/street/train/block_A/transforms.json
# cp datasets/MatrixCity/street/pose/block_A/transforms_test.json datasets/MatrixCity/street/test/block_A_test/transforms.json

# # Gather images and initialize sparse folder
# python tools/transform_json2txt_mc_street.py --source_path datasets/MatrixCity/street/train/block_A --intrinsic_path datasets/MatrixCity/street/pose/block_A/transforms_train.json
# python tools/transform_json2txt_mc_street.py --source_path datasets/MatrixCity/street/test/block_A_test --intrinsic_path datasets/MatrixCity/street/pose/block_A/transforms_test.json

# # Remove the old sparse folder and use the downloaded one
# rm -rf datasets/MatrixCity/street/train/block_A/sparse
# rm -rf datasets/MatrixCity/street/test/block_A_test/sparse

# mv data/colmap_results/matrix_city_street/train/sparse datasets/MatrixCity/street/train/block_A
# mv data/colmap_results/matrix_city_street/test/sparse datasets/MatrixCity/street/test/block_A_test