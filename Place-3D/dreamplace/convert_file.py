import Params
import PlaceDB

if __name__ == "__main__":
    logging.root.name = 'DREAMPlace'
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)
    params = Params.Params()
    
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    
    new_file = ""
    
    with ("example.input", 'r', encoding='utf-8') as file:
        
        file.write("")
        