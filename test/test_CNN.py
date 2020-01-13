import unittest
import" ../src/CNN_perso.py"


CREATE_CSV = True
PATH_DATA = "../DATA"
PATH_CAT = "/PetImages/Cat"
PATH_DOG = "/PetImages/Dog"

class TestCNN(unittest.TestCase):

    def test1(self):

        if CREATE_CSV:
            make_csv(PATH_DATA, [PATH_CAT,PATH_DOG])

    def test2(self):

        valid_ratio = 0.2  # Going to use 80%/20% split for train/valid

        train_test_dataset = CatDogDataset(csv_file='training_data.csv')

        nb_train = int((1.0 - valid_ratio) * len(train_test_dataset))
        nb_valid =  int(valid_ratio * len(train_test_dataset))
        train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_test_dataset, [nb_train, nb_valid])
    
    def testTensor(self):
        a =1


if __name__ == '__main__':
    unittest.main()