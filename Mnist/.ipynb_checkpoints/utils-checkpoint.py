from collections import defaultdict
import numpy as np
import matplotlib.pylab as plt

def predict(model, dataset):
    return model.predict(dataset).argmax(axis=-1)

def get_images_classes(dataset):

    images = np.concatenate([x for x, y in dataset], axis=0)
    classes = np.concatenate([y for x, y in dataset], axis=0)

    return images, classes

def display_image(image):
    """
    displays an image
    """
    plt.imshow(image.squeeze())

class Confusion:
    def __init__(self, model, dataset):
        """
        initializer
        """
        self.model = model
        self.images = np.concatenate([x for x, y in dataset], axis=0)
        self.classes = np.concatenate([y for x, y in dataset], axis=0)
        self.predictions = model.predict(dataset).argmax(axis=-1)
        self.map = self.create_hashmap()
        
    def create_hashmap(self):
        """
        creates hashmap data structure:
        map[true_class][predicted_class] = list of images
        """
        hashmap = defaultdict(lambda: defaultdict(list))
        for image, class_, prediction in zip(self.images, self.classes, self.predictions):
            hashmap[class_][prediction].append(np.array(image))
        return hashmap
    
    def print_confusion(self):
        """
        print basic confusion matrix numbers
        """
        classes = list(set(self.classes))
        for true_class in classes:
            for predicted_class in classes:
                print(len(self.map[true_class][predicted_class]),end=" ")
            print()
            
    def confusion_images(self):
        """
        display the confusion matrix with images
        """
        """
        plt.subplot(2,5,prediction+1)
    plt.axis("off")
    plt.title(prediction)
    plt.imshow(im_difference_pos.squeeze())
        """
        classes = list(set(self.classes))
        i = 1
        for true_class in classes:
            for predicted_class in classes:
                plt.subplot(len(classes),len(classes),i)
                plt.axis("off")
#                 plt.title(f"{class_1},{class_2}")
                average_image = self.get_mean_image(true_class, predicted_class)
                if average_image == []:
                    average_image = np.zeros((28, 28))
                
                plt.imshow(average_image.squeeze())
                i += 1
                                
    
    def average_image(self, images):
        """
        returns the average image from a list of images
        """
        images_numpy = np.array(images)
        mean_image = images_numpy.mean(axis=0)
        return mean_image
        
    def get_classified(self, true_class, predicted_class):
        """
        returns the images of class true_class that are predicted as predicted_class
        """
        images = self.map[true_class][predicted_class]
#         if images == []:
#             return -1
# #             raise Exception(f'There are no images of class {true_class} that were classified as {predicted_class}')
        return images
    
    def get_mean_image(self, true_class, predicted_class):
        """
        returns the mean image for images classified of class true_class that are predicted as predicted_class
        """
        images = self.get_classified(true_class, predicted_class)
        if images == []:
            return []
        return self.average_image(images)
        
    def get_difference(self, image_1, image_2):
        """
        returns the difference between two images
        """
        return np.subtract(image_1, image_2)
    
    def negate_negatives(self, image):
        """
        returns image after negating negative values
        """
        return np.where(image<0, 0, image)
    
    def get_positive_difference(self, image_1, image_2):
        """
        returns the difference between two images with negatives set to zero
        """
        image_difference = self.get_difference(image_1, image_2)
        image_positive = self.negate_negatives(image_difference)
        return image_positive
        
    def get_difference_classes(self, true_class_1, predicted_class_1, true_class_2, predicted_class_2):
        """
        returns the difference of the mean images of true_class_1 predicted as predicted_class_1 minus
        images of true_class_2 predicted as predicted_class_2
        """
        image_1 = self.get_mean_image(true_class_1, predicted_class_1)
        image_2 = self.get_mean_image(true_class_2, predicted_class_2)
        
        diff = self.get_difference(image_1, image_2)
        return diff
        
        
    def get_positive_difference_classes(self, true_class_1, predicted_class_1, true_class_2, predicted_class_2):
        """
        returns the positive difference of images of true_class_1 predicted as predicted_class_1 minus
        images of true_class_1 predicted as predicted_class_1
        """
        image_1 = self.get_mean_image(true_class_1, predicted_class_1)
        image_2 = self.get_mean_image(true_class_2, predicted_class_2)
        
        diff = self.get_positive_difference(image_1, image_2)
        return diff
    
    def get_false_positives(self, predicted_class):
        """
        returns a list of images that are falsely identified as predicted_class
        """
        classes = list(set(self.classes))
        images = []
        
        for true_class in classes:
            if true_class == predicted_class:
                continue
            images.extend(self.map[true_class][predicted_class])
            
        return images
    
    def get_mean_false_positives(self, predicted_class):
        """
        returns the mean image of those that are falsely identified as predicted_class
        """
        images = self.get_false_positives(predicted_class)
        return self.average_image(images)
    
    def get_false_negatives(self, true_class):
        """
        returns a list of images of type true_class that are falsely predicted
        """
        classes = list(set(self.classes))
        images = []
        for predicted_class in classes:
            if true_class == predicted_class:
                continue
            images.extend(self.map[true_class][predicted_class])
            
        return images
    
    def get_mean_false_negatives(self, true_class):
        """
        returns the mean image of type true_class that are falsely predicted
        """
        images = self.get_false_negatives(true_class)
        return self.average_image(images)
    
    def get_true_positives(self, class_):
        """
        returns a list of images of type class_ that are correctly predicted
        """
        return self.map[class_][class_]
    
    def get_mean_true_positives(self, class_):
        """
        returns the mean image of type class_ that are correctly predicted
        """
        images = self.get_true_positives(class_)
        return self.average_image(images)
    