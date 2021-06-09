#!/bin/bash
from plan2scene.config_manager import ConfigManager
import argparse
import os
import os.path as osp
import glob
import subprocess
import logging

if __name__ == "__main__":
    """
    Preview GNN preditions for every nth epoch. 
    All surfaces (including observed surfaces) are assigned a propagated texture computed using texture embeddings of neighboring surfaces.
    """
    conf = ConfigManager()
    parser = argparse.ArgumentParser(
        description="Preview GNN preditions for every nth epoch. All surfaces are assigned the GNN prediction (including observed surfaces).")
    conf.add_args(parser)
    parser.add_argument("output_path", type=str, help="Output directory which will contain the preview.html file.")
    parser.add_argument("interval", type=int, help="Interval between previewed epochs.")
    parser.add_argument("gnn_train_path", type=str,
                        help="Path to GNN train directory. This directory has the conf directory and checkpoints directory.")
    args = parser.parse_args()
    conf.process_args(args, output_is_dir=True)

    output_path = args.output_path
    gnn_train_path = args.gnn_train_path
    interval = args.interval
    conf_path = osp.join(gnn_train_path, "conf", "texture_prop.json")
    checkpoints_path = osp.join(gnn_train_path, "checkpoints")

    assert osp.exists(conf_path)
    assert osp.exists(checkpoints_path)

    current_checkpoint_id = interval
    preview_images = None

    with open(osp.join(output_path, "preview.html"), "w") as f:
        f.write("<table>\n")
        while True:
            candidates = glob.glob(osp.join(checkpoints_path, f"loss-*-epoch-{current_checkpoint_id}.ckpt"))
            if len(candidates) == 0:
                break
            candidate = candidates[0]
            logging.info(f"Processing {candidate}")

            command = "/bin/bash ./code/scripts/plan2scene/texture_prop/preview_all_prop.sh {output_path} {conf_path} {checkpoint_path}".format(
                output_path=osp.join(output_path, f"epoch_{current_checkpoint_id}"),
                conf_path=conf_path,
                checkpoint_path=candidate
            )
            my_env = os.environ.copy()
            assert subprocess.call(command, shell=True, env=my_env) == 0

            # Preview
            if preview_images is None:
                preview_images = os.listdir(osp.join(output_path, f"epoch_{current_checkpoint_id}", "archs"))
                preview_images = [a for a in preview_images if a.endswith(".png")]
                f.write("<tr>\n")
                f.write("<th>Epoch</th>\n")
                for preview_image in preview_images:
                    f.write(f"<th>{preview_image}</th>\n")

                f.write("</tr>\n")

            f.write("<tr>\n")
            f.write(f"<td>{current_checkpoint_id}</td>\n")

            for preview_image in preview_images:
                f.write("<td>\n")
                f.write("<img src='{img_path}' style='width:200px;'/>".format(img_path=osp.join(f"epoch_{current_checkpoint_id}", "archs", preview_image)))
                f.write("</td>\n")

            f.write("</tr>\n")
            f.flush()

            current_checkpoint_id += interval
        f.write("</table>\n")

