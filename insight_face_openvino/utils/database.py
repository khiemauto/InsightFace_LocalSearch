from . import io_utils
import pickle
import os
import numpy as np
import cv2
import datetime
from .io_utils import read_image
from ..sdk import FaceRecognitionSDK

import shutil

class FaceRecognitionSystem:

    """A simple class that demonstrates how Face SDK can be integrated into other systems."""

    def __init__(self, photo_dir: str, sdk_config: dict = None):
        self.photoid_to_username_photopath = {}
        self.sdk = FaceRecognitionSDK(sdk_config)

        self.descriptors_db_filename = "descriptors.index"
        self.photoid_to_username_photopath_filename = "id_to_username.pkl"
        self.photo_dir = photo_dir

    def get_user_name(self, photo_id: int) -> str:
        return self.photoid_to_username_photopath[photo_id][0]
    
    def get_photo_path(self, photo_id: int) -> str:
        return self.photoid_to_username_photopath[photo_id][1]

    def add_photo_by_user_name(self, image: np.ndarray, user_name: str):
        ''' Add RGB photo for user name
        image: RGB image
        user_name: Name user
        '''
        try:
            photo_id = max(self.photoid_to_username_photopath) + 1 if len(self.photoid_to_username_photopath) > 0 else 0
            self.add_photo_by_photo_id(image, photo_id)
            path = os.path.join(self.photo_dir, user_name)
            if not os.path.exists(path):
                os.makedirs(path)
            filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
            photo_path = os.path.join(self.photo_dir, user_name, filename)
            io_utils.save_image(image, photo_path)
            self.photoid_to_username_photopath[photo_id] = [user_name, photo_path]
        except:
            return False, None, None
        return True, photo_id, photo_path

    def add_photo_descriptor_by_user_name(self, image, descriptor, user_name: str):
        ''' Add RGB photo and descriptor for user name
        image: RGB image
        descriptor : 512D embed
        user_name: Name user
        '''
        try:
            photo_id = max(self.photoid_to_username_photopath) + 1 if len(self.photoid_to_username_photopath) > 0 else 0
            self.add_descriptor_by_photo_id(descriptor, photo_id)
            path = os.path.join(self.photo_dir, user_name)
            if not os.path.exists(path):
                os.makedirs(path)
            filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
            photo_path = os.path.join(self.photo_dir, user_name, filename)
            io_utils.save_image(image, photo_path)
            self.photoid_to_username_photopath[photo_id] = [user_name, photo_path]
        except:
            return False, None, None
        return True, photo_id, photo_path

    def del_photo_by_user_name(self, user_name: str):
        ''' Remove all photo for user name
        user_name: Name user
        '''
        nbphoto = 0
        ret = False
        for photoid, (username, photopath) in list(self.photoid_to_username_photopath.items()):
            if username == user_name:
                self.sdk.delete_photo_by_photo_id(photoid)
                del self.photoid_to_username_photopath[photoid]
                nbphoto +=1
                ret = True
        
        path = os.path.join(self.photo_dir, user_name)
        if os.path.exists(path):
            shutil.rmtree(path)

        return ret, nbphoto

    def add_photo_by_photo_id(self, image: np.ndarray, photo_id: int) -> None:
        """Add new photo for user name
        photo_path: path image
        user_name: Name user
        """
        self.sdk.add_photo_by_photo_id(image, photo_id)

    def add_descriptor_by_photo_id(self, descriptor, photo_id: int) -> None:
        """Add new descriptor for user name
        photo_path: path image
        user_name: Name user
        """
        self.sdk.add_descriptor(descriptor, photo_id)

    def delete_photo_by_photo_id(self, photo_id: int) -> bool:
        """Removes photo (descriptor) from the database.
        photo_id: id of the photo in the database.
        """
        try:
            self.sdk.delete_photo_by_photo_id(photo_id)
            photopath = self.photoid_to_username_photopath[photo_id][1]
            os.remove(photopath)
            del self.photoid_to_username_photopath[photo_id]
        except:
            return False
        else:
            return True

    def create_database_from_folders(self, root_path: str) -> None:
        """Create face database from hierarchy of folders.
        Each folder named as an individual and contains his/her photos.
        """

        try:
            photo_id = max(self.photoid_to_username_photopath) + 1 if len(self.photoid_to_username_photopath) > 0 else 0
            for username in os.listdir(root_path):

                user_images_path = os.path.join(root_path, username)

                if not os.path.isdir(user_images_path):
                    continue

                # iterating over user photos
                for filename in os.listdir(user_images_path):
                    print(f"Adding {filename} from {user_images_path}")
                    photo_path = os.path.join(root_path, username, filename)
                    photo = read_image(photo_path)
                    self.sdk.add_photo_by_photo_id(photo, photo_id)
                    self.photoid_to_username_photopath[photo_id] = [username, photo_path]
                    photo_id += 1
        except Exception:
            self.photoid_to_username_photopath = {}
            self.sdk.reset_database()
            raise

    def save_database(self, folder_path: str) -> None:

        descriptors_path = os.path.join(folder_path, self.descriptors_db_filename)
        id_to_user_path = os.path.join(folder_path, self.photoid_to_username_photopath_filename)

        # save descriptors to userid index (sdk)
        self.sdk.save_database(descriptors_path)

        # save our own id to username mapping
        with open(id_to_user_path, "wb") as fp:
            pickle.dump(self.photoid_to_username_photopath, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load_database(self, folder_path: str) -> None:

        descriptors_path = os.path.join(folder_path, self.descriptors_db_filename)
        id_to_user_path = os.path.join(folder_path, self.photoid_to_username_photopath_filename)

        # load descriptors to userid index (sdk)
        self.sdk.load_database(descriptors_path)

        # load our own id to username mapping
        with open(id_to_user_path, "rb") as fp:
            self.photoid_to_username_photopath = pickle.load(fp)