from molbloom import buy

class score_exist:
    def __init__(self, catalogs = ['zinc20', 'zinc-instock', 'zinc-instock-mini', 'surechembl']) -> None:
        self.catalogs = catalogs
    
    def check(self, SMILES):
        if type(SMILES) == str:
            buyornot = []
            for database in self.catalogs:
                in_database = buy(SMILES, catalog=database)
                buyornot.append(in_database)
            if True in buyornot:
                return 1
            else:
                return 0
        else:
            raise TypeError('SMILES should be a string')

if __name__ == "__main__":
    score = score_exist()
    print(score.check('CCCC'))