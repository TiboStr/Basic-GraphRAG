entity_types = ["organization", "person", "geo", "event"]
tuple_delimiter = "|"  # MSFT GraphRAG uses <|>, but I noticed simpler models mess this delimiter up, so we'll use a simpler one
record_delimiter = "##"
completion_delimiter = "<|COMPLETE|>"
