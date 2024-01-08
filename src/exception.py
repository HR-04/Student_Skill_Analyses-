import sys
from src.logger import logging

def error_message_details(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error Occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error))
        
    return error_message
    
    

class CustomerException(Exception):
    def __init__(self,error_message,error_deatil:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(error_message,error_detail=error_deatil)
        
    def __str__(self):
        return self.error_message
    
    
if __name__ == "__main__":
    
    try:
        a=1/0
    except Exception as e:
        logging.info("Divide by Zero Error")
        raise CustomerException(e,sys)