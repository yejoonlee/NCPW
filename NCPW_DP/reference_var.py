class reference():

    def __init__(self):
        self.classification_model_path = '/Users/yeznable/Documents/GitHub/NCPW/NCPW_DL/classify_log/GRU_model_240'
        self.classification_sequence = 4
        self.update_rate = 0.5

    def get_ref_class(self):
        return self.classification_sequence, self.classification_model_path

    def get_agent_path(self, id:str):
        return f'/Users/yeznable/Documents/GitHub/NCPW/agents/DQN_agent_{id}.ptb'

    def get_ref_update(self):
        return self.update_rate