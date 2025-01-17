import yaml


class parse(object):
    """
    This class reads yaml parameter file and allows dictionary like access to the members.
    """

    def __init__(self, path):  # 读取
        with open(path, 'r') as file:
            self.parameters = yaml.safe_load(file)

    # Allow dictionary like access
    def __getitem__(self, key):  # 访问
        return self.parameters[key]

    def save(self, filename):  # 保存
        with open(filename, 'w') as f:
            yaml.dump(self.parameters, f)
