import sys

def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    return (
        f"Error occurred in script [{file_name}] "
        f"at line number [{exc_tb.tb_lineno}] "
        f"with message [{str(error)}]"
    )


class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message