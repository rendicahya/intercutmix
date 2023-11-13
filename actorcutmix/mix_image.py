import cv2

if __name__ == "__main__":
    image1 = cv2.imread(
        "/nas.dbms/randy/datasets/ucf101-frames/Basketball/v_Basketball_g01_c01/img_00001.jpg"
    )

    image2 = cv2.imread(
        "/nas.dbms/randy/datasets/ucf101-frames/Basketball/v_Basketball_g05_c01/img_00001.jpg"
    )

    actor_mask1 = cv2.imread(
        "/nas.dbms/randy/datasets/ucf101-bbox-mask/Basketball/v_Basketball_g01_c01/0000.png",
        cv2.IMREAD_GRAYSCALE,
    )

    actor_mask2 = cv2.imread(
        "/nas.dbms/randy/datasets/ucf101-bbox-mask/Basketball/v_Basketball_g05_c01/0000.png",
        cv2.IMREAD_GRAYSCALE,
    )

    scene_mask1 = 255 - actor_mask1
    scene_mask2 = 255 - actor_mask2

    actor1 = cv2.bitwise_and(image1, image1, mask=actor_mask1)
    actor2 = cv2.bitwise_and(image2, image2, mask=actor_mask2)

    scene1 = cv2.bitwise_and(image1, image1, mask=scene_mask1)
    scene2 = cv2.bitwise_and(image2, image2, mask=scene_mask2)

    mix1 = actor1 + cv2.bitwise_and(scene2, scene2, mask=scene_mask1)
    mix2 = actor2 + cv2.bitwise_and(scene1, scene1, mask=scene_mask2)

    cv2.imwrite("images/actor1.jpg", actor1)
    cv2.imwrite("images/actor2.jpg", actor2)

    cv2.imwrite("images/scene1.jpg", scene1)
    cv2.imwrite("images/scene2.jpg", scene2)

    cv2.imwrite("images/actor_mask1.jpg", actor_mask1)
    cv2.imwrite("images/actor_mask2.jpg", actor_mask2)

    cv2.imwrite("images/scene_mask1.jpg", scene_mask1)
    cv2.imwrite("images/scene_mask2.jpg", scene_mask2)

    cv2.imwrite("images/mix1.jpg", mix1)
    cv2.imwrite("images/mix2.jpg", mix2)
