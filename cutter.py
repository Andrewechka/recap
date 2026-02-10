def cut_webtoon_cascade(...):
    ...
    scenes_xy = _merge_close_and_small(scenes_xy, cfg, total_h=H)

    # New logic to refine scenes
    high_scenes = [scene for scene in scenes_xy if some_condition_for_high(scene)]
    refined_scenes = split_scene_adaptive(high_scenes)
    final_imgs = generate_final_images(refined_scenes)

    return final_imgs