class Preprocessor():
    def __init__(self, remove_background: bool = True, dim_x: int = 200, dim_y: int = 200):
        """
        constructs the Preprocessor

        :param remove_background: if true, image background is removed
        :param dim_x: number of pixels in x dimension
        :param dim_y: number of pixels in y dimension
        """
        self.remove_background = remove_background
        self.dimensions = (dim_x, dim_y)

    def __call__(self, image_path: str):
        """
        Takes the path to an image and returns the preprocessed version of the image

        :param image_path: path to the image that should be processed
        :return: the preprocessed image
        """
        print(f'format image {image_path}, background removed?{self.remove_background}')
        raise NotImplementedError("this has not yet been implemented")
        # here you should return the preprocessed image
        return None


if __name__ == "__main__":
    img_path="select/an/image"
    test_processor = Preprocessor(remove_background=True)
    processed_image = test_processor(img_path)
