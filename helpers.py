import re

POST_EDA_TYPES = ['continuous', 'ordinal', 'binary', 'nominal']
IDENTIFIER_PATTERN = r'[a-zA-Z][a-zA-Z0-9_]*' 
TYPE_PATTERN = r'(?:binary|continuous|ordinal|nominal|binary_nominal|binary_ordinal|time)'

def parse_feature_metadata(col: str):
    """
    Given a column name, returns a dictionary containing the following:
    qid: The ID of the survey question associated with this feature
    name: Description of the feature itself
    type: The type of variable this feature is (continuous, ordinal or binary)
    dummy: The dummy column number of this feature if it is a dummy column
    The column names of the data were changed manually to follow a comfortable format that allows
    easy splitting into the different aspects of the feature. 
    """
    if not re.fullmatch(fr'^Q\d+__{IDENTIFIER_PATTERN}__{TYPE_PATTERN}(__\d+)?$', col):
        return None
    parts = col.split('__', maxsplit=3)
    qid, name, ftype = parts[:3]
    dummy = parts[3] if len(parts)==4 else None
    return {"qid": qid, "name": name, "type": ftype, "dummy": dummy}

if __name__=="__main__":
    pass
