def cut_webtoon_cascade(...):
    # existing logic

    # Adaptive scene refinement
    for scene in scenes:
        if scene.height > SOME_HEIGHT_THRESHOLD:
            scene = split_scene_adaptive(scene)

    return scenes