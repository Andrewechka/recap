def cut_webtoon_cascade(...):
    ...
    scenes_xy = _merge_close_and_small(scenes_xy, cfg, total_h=H)
    # New adaptive scene splitting logic for high scenes
    # Implement the refined logic here that creates an adaptive scene splitting for high scenes
    final_imgs = some_logic_to_get_final_imgs(scenes_xy)
    return final_imgs
