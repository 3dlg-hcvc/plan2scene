import argparse
import os
import os.path as osp
import logging
import subprocess
from plan2scene.config_manager import ConfigManager

EXPORT_COMMAND_SPEC = "{node_path} {script_path} --input_type path --config_file {config_file} --output_format {output_format} --input {input_path} --output_dir {output_dir} {additional_args}"
RENDER_COMMAND_SPEC = "{node_path} {script_path} --input {input_path} --output_dir {output_dir} --config_file {config_file} {additional_args}"

if __name__ == "__main__":
    """
    Renders arch.json/scene.json files in a specified directory. 
    """

    conf = ConfigManager()
    parser = argparse.ArgumentParser(description="Scan for .arch.json files and render them as .arch.png files.")
    conf.add_args(parser)
    parser.add_argument("search_path", help="Path to walk for .arch.json files")
    parser.add_argument("--scene-json", default=False, action="store_true", help="Process scene.json files instead of arch.json files.")
    parser.add_argument("--export", default=False, action="store_true", help="Export meshes in GLTF format.")

    args = parser.parse_args()
    conf.process_args(args)

    search_path = args.search_path
    process_scene_json = args.scene_json
    export = args.export

    print("Searching in: %s" % (search_path))

    if process_scene_json:
        extension = ".scene.json"
    else:
        extension = ".arch.json"

    found_files = []
    for root, dirs, files in os.walk(search_path):
        files = [a for a in files if a.endswith(extension)]
        for file in files:
            if not osp.exists(osp.join(root, file.replace(extension, ".arch.png"))):
                found_files.append(osp.join(root, file))

    print("Found %d" % (len(found_files)))

    found_files = sorted(found_files)

    for i, file in enumerate(found_files):
        print("[%d/%d] Processing %s" % (i, len(found_files), file))
        output_path = osp.dirname(file)
        render_command = None
        export_command = None
        script_path = None

        if export:
            # Export
            script_path = conf.render_config.export.stk_export_script_path
            if process_scene_json:
                # Scene json
                config_file = conf.render_config.export.scene_json.stk_config_file_path
                additional_args = conf.render_config.export.scene_json.extra_args
            else:
                # Arch json
                config_file = conf.render_config.export.arch_json.stk_config_file_path
                additional_args = conf.render_config.export.arch_json.extra_args
        else:
            # Render
            script_path = conf.render_config.render.stk_render_script_path
            if process_scene_json:
                # Scene json
                config_file = conf.render_config.render.scene_json.stk_config_file_path
                additional_args = conf.render_config.render.scene_json.extra_args
            else:
                # Arch json
                additional_args = conf.render_config.render.arch_json.extra_args
                config_file = conf.render_config.render.arch_json.stk_config_file_path

        if export:
            command = EXPORT_COMMAND_SPEC.format(
                node_path=conf.render_config.node_path,
                script_path=script_path,
                config_file=config_file,
                output_format=conf.render_config.export.scene_json.output_format,
                input_path=osp.abspath(file),
                output_dir=osp.dirname(file),
                additional_args=" ".join(additional_args)
            )
        else:
            command = RENDER_COMMAND_SPEC.format(
                node_path=conf.render_config.node_path,
                script_path=script_path,
                input_path=osp.abspath(file),
                output_dir=osp.dirname(file),
                config_file=config_file,
                additional_args=" ".join(additional_args)
            )

        assert subprocess.call(command, shell=True) == 0

