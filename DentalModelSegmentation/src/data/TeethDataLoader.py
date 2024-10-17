import torch.utils.data as data
import numpy as np
import os
import json
from src.utils.input import read_obj


def _read_json_label(path):
    """
    Read ground truth label from given JSON file

    The JSON object contains keys ["id_patient", "jaw", "labels", "instances"],
    only the JSON array "labels" is what we need.

    :param path: The path of GT JSON file
    :type path: str
    :return: Class labels of the case
    :rtype: list
    """
    file = open(path, 'r')
    json_data = json.load(file)
    return json_data['labels']


class TeethDataset(data.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.model_dir = os.path.join(data_dir, '3D_scans_per_patient_obj_files')
        self.gt_dir = os.path.join(data_dir, 'ground-truth_labels_instances')

        # Get all models in model_dir
        all_patients = os.listdir(self.model_dir)

        all_ids, all_jaws, all_data, all_labels = [], [], [], []

        for patient in all_patients:
            for jaw in ['upper', 'lower']:
                # For data in the challenge, the obj file is named "{patient_id}/{patient_id}_{upper/lower}.obj"
                print('Dataloader {} - {}'.format(patient, jaw))
                # vertices of the obj, faces are not considered
                vertices, _, normals =\
                    read_obj(os.path.join(self.model_dir, patient, '%s_%s.obj' % (patient, jaw)), True)

                labels = _read_json_label(os.path.join(self.gt_dir, patient, '%s_%s.json' % (patient, jaw)))
                labels = np.array(labels)

                all_ids.append(patient)
                all_jaws.append(jaw)
                all_data.append(np.concatenate((vertices, normals), axis=1))
                all_labels.append(labels)

        self.all_ids, self.all_jaws, self.all_data, self.all_labels = \
            all_ids, all_jaws, all_data, all_labels

    def __getitem__(self, index):
        return self.all_ids[index], self.all_jaws[index], \
               self.all_data[index], self.all_labels[index]

    def __len__(self):
        return len(self.all_data)
