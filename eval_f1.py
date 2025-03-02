

import json
import copy
import os
import numpy as np
import open3d as o3d
import argparse
def EvaluateHisto(
    source,
    target,
    threshold,
    plot_stretch,
):

    print("[compute_point_cloud_to_point_cloud_distance]")
    distance1 = source.compute_point_cloud_distance(target)
    print("[compute_point_cloud_to_point_cloud_distance]")
    distance2 = target.compute_point_cloud_distance(source)

    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = get_f1_score_histo2(threshold, plot_stretch, distance1,
                            distance2)



    return [
            precision,
            recall,
            fscore,
            edges_source,
            cum_source,
            edges_target,
            cum_target,
    ]



def get_f1_score_histo2(threshold,
                        plot_stretch,
                        distance1,
                        distance2,
                        ):
    print("[get_f1_score_histo2]")
    dist_threshold = threshold
    if len(distance1) and len(distance2):

        recall = float(sum(d < threshold for d in distance2)) / float(
            len(distance2))
        precision = float(sum(d < threshold for d in distance1)) / float(
            len(distance1))
        fscore = 2 * recall * precision / (recall + precision)
        num = len(distance1)
        bins = np.arange(0, dist_threshold * plot_stretch, dist_threshold / 100)
        hist, edges_source = np.histogram(distance1, bins)
        cum_source = np.cumsum(hist).astype(float) / num

        num = len(distance2)
        bins = np.arange(0, dist_threshold * plot_stretch, dist_threshold / 100)
        hist, edges_target = np.histogram(distance2, bins)
        cum_target = np.cumsum(hist).astype(float) / num

    else:
        precision = 0
        recall = 0
        fscore = 0
        edges_source = np.array([0])
        cum_source = np.array([0])
        edges_target = np.array([0])
        cum_target = np.array([0])

    return [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ]




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COLMAP Python Interface')
    # 添加参数
    parser.add_argument('--ply_path_pred', type=str, required=True, help='Path to the database file')
    parser.add_argument('--ply_path_gt', type=str, required=True, help='Path to the database file')
    parser.add_argument('--dtau', type=float, required=True)
    args = parser.parse_args()

    print(args.ply_path_pred)
    mesh = o3d.io.read_triangle_mesh(args.ply_path_pred)
    mesh.remove_unreferenced_vertices()
    # pcd = mesh.sample_points_uniformly(number_of_points=12800000)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
    # pcd = o3d.io.read_point_cloud(ply_path)

    print(args.ply_path_gt)
    mesh_gt = o3d.io.read_triangle_mesh(args.ply_path_gt)
    mesh_gt.remove_unreferenced_vertices()
    # pcd = mesh.sample_points_uniformly(number_of_points=12800000)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(mesh_gt.vertices)

    dTau = args.dtau

    plot_stretch = 5
    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = EvaluateHisto(
        pcd,
        pcd_gt,
        dTau / 2.0,
        plot_stretch,

    )
    eva = [precision, recall, fscore]
    print("==============================")
    print("==============================")
    print("distance tau : %.3f" % dTau)
    print("precision : %.4f" % eva[0])
    print("recall : %.4f" % eva[1])
    print("f-score : %.4f" % eva[2])
    print("==============================")