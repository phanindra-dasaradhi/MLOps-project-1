# import sys

# class CustomException(Exception):
#     def __init__(self, error_message, error_details):
#         message = str(error_message)
#         detailed = self.get_detailed_error_message(message, error_details)
#         super().__init__(detailed)
#         self.error_message = detailed

#     @staticmethod
#     def get_detailed_error_message(error_message, error_details):
#         exc_type, exc_value, exc_tb = error_details  # tuple unpack

#         if exc_tb is None:
#             return f"Error: {error_message}"

#         file_name = exc_tb.tb_frame.f_code.co_filename
#         line_number = exc_tb.tb_lineno
#         return f"Error in {file_name} at line {line_number}: {error_message}"

#     def __str__(self):
#         return self.error_message



import sys

class CustomException(Exception):
    def __init__(self, error_message, error_details):
        message = str(error_message)
        detailed = self.get_detailed_error_message(message, error_details)
        super().__init__(detailed)
        self.error_message = detailed

    @staticmethod
    def get_detailed_error_message(error_message, error_details):
        # We check if error_details has the exc_info attribute (is the sys module)
        # If not, we fall back to the standard sys module imported at the top
        if hasattr(error_details, 'exc_info'):
            _, _, exc_tb = error_details.exc_info()
        else:
            _, _, exc_tb = sys.exc_info()

        if exc_tb is None:
            return f"Error: {error_message}"

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error in {file_name} at line {line_number}: {error_message}"

    def __str__(self):
        return self.error_message