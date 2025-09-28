class ProcessInterrupted(Exception):
    def __init__(self, data_value=None, message="Process Interrupted by the user"):
        self.data_value = data_value
        super().__init__(message)