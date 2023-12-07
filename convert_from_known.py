import os
import numpy as np
import argparse
import shutil
import logging
import json

from scipy.spatial.transform import Rotation as R

from scene.colmap_loader import *
from scene.colmap_database import *
from scene.dataset_readers import storePly

CAMERA_ID = 1

def create_database(database_path):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    return db

def create_intrinsics(intrinsic):
    W = 512#intrinsic['columns'] if 'columns' in intrinsic else intrinsic['width']
    H = 512#intrinsic['rows'] if 'rows' in intrinsic else intrinsic['height']
    if 'fov' in intrinsic:
        f = W / 2 / np.tan(intrinsic['fov'] / 2 / 180 * np.pi)
        fx = f
        fy = f
        cx = W / 2
        cy = H / 2
    else:
        fx = intrinsic['fx']
        fy = intrinsic['fy']
        cx = intrinsic['cx']
        cy = intrinsic['cy']
    K = np.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K

def load_cam(filename):
    with open(filename, 'r') as f:
        cam = json.load(f)
    
    intr = cam['intrinsics']
    
    pose = np.array(cam['transform_matrix'])
    
    K = create_intrinsics(intr)

    F = np.array([[ 0,  1,  0 ], [ 0,  0, -1 ], [ 1,  0,  0 ]], dtype=np.float32)
    F = np.linalg.inv(F)
    F_align = np.eye(4)
    F_align[:3, :3] = F
    return pose @ F_align, K

def main():
    args = argparse.ArgumentParser("Colmap converter from known camera path")
    args.add_argument("--no_gpu", action='store_true')
    args.add_argument("--skip_matching", action='store_true')
    args.add_argument("--source_path", "-s", required=True, type=str)
    args.add_argument("--camera", default="OPENCV", type=str)
    args.add_argument("--colmap_executable", default="", type=str)
    args.add_argument("--resize", action="store_true")
    args.add_argument("--magick_executable", default="", type=str)

    args = args.parse_args()
    colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
    magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
    use_gpu = 1 if not args.no_gpu else 0

    if not args.skip_matching:
        os.makedirs(args.source_path + "/distorted/sparse/0", exist_ok=True)

    
        camfile = os.path.join(args.source_path, 'camera/circleClockwise_000001_camera.json')
        _, K = load_cam(camfile)
        fx, fy = K[0][0], K[1][1]
        cx, cy = K[0][2], K[1][2]

        image_index = 0
        fimg = open(os.path.join(args.source_path, 'distorted/sparse/0/images.txt'), 'w')
        fpts = open(os.path.join(args.source_path, 'distorted/sparse/0/points3D.txt'), 'w')


        with open(os.path.join(args.source_path, 'distorted/sparse/0/cameras.txt'), 'w') as fcam:
            fcam.write(str(CAMERA_ID) + ' ')
            fcam.write(CAMERA_MODEL_IDS[CAMERA_ID].model_name + ' ')
            fcam.write(str(int(2 * cx)) + ' ' + str(int(2 * cy)) + ' ')
            fcam.write(str(fx) + ' ' + str(fy) + ' ')
            fcam.write(str(cx) + ' ' + str(cy))
        fcam.close()

        # create database
        db = COLMAPDatabase.connect(os.path.join(args.source_path, 'distorted/database.db'))
        db.create_tables()
        camera_id = db.add_camera(CAMERA_ID, int(2 * cx), int(2 * cy), np.array([fx, fy, cx, cy]))

        for i, frame in enumerate(sorted(os.listdir(os.path.join(args.source_path, 'input')))):    
            img_file = os.path.join(args.source_path, 'input', frame)
            cam_file = os.path.join(args.source_path, 'camera', frame.replace("color.png", "camera.json"))

            # img = cv2.imread(img_file)
            # if img is None:
            #     continue
            # h, w, _ = img.shape
            # img_resize = cv2.resize(img, dsize=(w//args.downSample, h//args.downSample))
            # cv2.imwrite(os.path.join(args.project_name, 'images/{:06d}.jpg'.format(i)), img_resize)

            c2w, K = load_cam(cam_file)
            

            R_mat = np.linalg.inv(c2w[:3, :3])
            tvec = -R_mat @ c2w[:3, 3]


            # r = R.from_matrix(R_mat)
            # qvec = r.as_quat()

            qvec = rotmat2qvec(R_mat)
            image_index += 1
            fimg.write(
                str(image_index) + ' ' + \
                str(qvec[0].item()) + ' ' + \
                str(qvec[1].item()) + ' ' + \
                str(qvec[2].item()) + ' ' + \
                str(qvec[3].item()) + ' ' + \
                str(tvec[0].item()) + ' ' + \
                str(tvec[1].item()) + ' ' + \
                str(tvec[2].item()) + ' ' + \
                str(CAMERA_ID) + ' ' + frame + '\n'
            )
            fimg.write('\n')

            db.add_image(
                name=frame,
                camera_id=camera_id,
                prior_q=qvec,
                prior_t=tvec,
                image_id=image_index
            )

        fimg.close()
        fpts.close()

        db.commit()
        db.close()

        feature_extraction_cmd = "colmap feature_extractor "\
            "--database_path " + args.source_path + "/distorted/database.db \
            --image_path " + args.source_path + "/input \
            --ImageReader.single_camera 1 \
            --ImageReader.camera_model " + "PINHOLE" + " \
            --SiftExtraction.use_gpu " + str(use_gpu)
        exit_code = os.system(feature_extraction_cmd)
        if exit_code != 0:
            logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
            exit(exit_code)

        ## Feature matching
        feat_matching_cmd = colmap_command + " exhaustive_matcher \
            --database_path " + args.source_path + "/distorted/database.db \
            --SiftMatching.use_gpu " + str(use_gpu)
        exit_code = os.system(feat_matching_cmd)
        if exit_code != 0:
            logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
            exit(exit_code)

        
        triangulator_cmd = colmap_command + " point_triangulator " \
            "--database_path " + args.source_path  + "/distorted/database.db \
            --image_path " + args.source_path + "/input \
            --input_path " + args.source_path + "/distorted/sparse/0 \
            --output_path " + args.source_path + "/distorted/sparse \
            --Mapper.ba_global_function_tolerance=0.000001"
        # mapper_cmd = (colmap_command + " mapper \
        # --database_path " + args.source_path + "/distorted/database.db \
        # --image_path "  + args.source_path + "/input \
        # --output_path "  + args.source_path + "/distorted/sparse \
        # --Mapper.ba_global_function_tolerance=0.000001")
        exit_code = os.system(triangulator_cmd)
        if exit_code != 0:
            logging.error(f"Mapper failed with code {exit_code}. Exiting.")
            exit(exit_code)


    ### Image undistortion
    ## We need to undistort our images into ideal pinhole intrinsics.
    # img_undist_cmd = (colmap_command + " image_undistorter \
    #     --image_path " + args.source_path + "/input \
    #     --input_path " + args.source_path + "/distorted/sparse/0 \
    #     --output_path " + args.source_path + "\
    #     --output_type COLMAP")
    # exit_code = os.system(img_undist_cmd)
    # if exit_code != 0:
    #     logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    #     exit(exit_code)

    # files = os.listdir(args.source_path + "/sparse")
    # os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
    # # Copy each file from the source directory to the destination directory
    # for file in files:
    #     if file == '0':
    #         continue
    #     source_file = os.path.join(args.source_path, "sparse", file)
    #     destination_file = os.path.join(args.source_path, "sparse", "0", file)
    #     shutil.move(source_file, destination_file)
    
    # save sparse point cloud
    os.makedirs(args.source_path + "/point_cloud", exist_ok=True)
    xyzs, rgbs, _ = read_points3D_binary(os.path.join(args.source_path, "distorted/sparse", "points3D.bin"))
    storePly(os.path.join(args.source_path, "point_cloud/point_cloud.ply"), xyzs, rgbs)


if __name__ == "__main__":
    main()
    # cmd = 'colmap feature_extractor '
    # cmd += '--database_path 0000/database.db '
    # cmd += '--image_path 0000/images'

    # # feature extraction
    # os.system(cmd)

    # cmd = 'colmap exhaustive_matcher '
    # cmd += '--database_path 0000/database.db'

    # os.system(cmd)
    # s = os.path.join('data/carla_v1/distorted/sparse/points3D.bin')
    # xyzs, rgbs, _ = read_points3D_binary(s)
    # storePly('data/carla_v1/point_cloud/test.ply', xyzs, rgbs)

    # img_file = 'data/carla_v1/distorted/sparse/images.bin'
    # img = read_extrinsics_binary(img_file)
    # print(img[1])
    # print(img[2])