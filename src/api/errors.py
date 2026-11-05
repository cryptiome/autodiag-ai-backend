class ApiError(Exception):
    def __init__(self, code, message, status_code=400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = int(status_code)

    def to_dict(self):
        return {
            'error': {
                'code': self.code,
                'message': self.message,
            }
        }
