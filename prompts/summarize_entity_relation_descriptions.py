SUMMARIZE_ENTITIES = """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given and entities, and a list of descriptions, all related to the same entity.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have the full context.

#######
-Data-
Entity: {entity_name}
Description List: {description_list}
#######
Output:
"""

SUMMARIZE_RELATIONSHIPS = """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given a pair of related entities, and a list of descriptions, all related to the same relationship.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have the full context.

#######
-Data-
Relation: {entity_name}
Description List: {description_list}
#######
Output:
"""
