entity_types = [
    # General Types
    "organization",  # Companies, departments, or teams involved
    "person",  # Individuals responsible for tasks or oversight
    "geo",  # Locations, facilities, or geographic areas
    "event",  # Maintenance events, inspections, incidents

    # Domain-specific types
    "equipment",  # Machines, tools, or systems mentioned
    "component",  # Specific parts of equipment or machinery
    "procedure",  # Specific maintenance or safety procedures
    "schedule",  # Dates, times, or cycles for maintenance
    "issue",  # Problems, faults, or errors noted
    "material",  # Spare parts, consumables, or resources used
    "hazard",  # Safety hazards or risk factors
    "standard",  # Industry or company standards, guidelines
    "contractor",  # External service providers or contractors
    "document",  # Manuals, checklists, or procedural documents
    "software",  # Maintenance management or monitoring systems
    "sensor",  # Monitoring devices, IoT components
    "measurement",  # Specific metrics, readings, or thresholds
    "license",  # Certifications, permissions, or compliance documents
    "regulation",  # Government or industry regulations
    "inspection",  # Scheduled or unscheduled checks
    "location",  # Specific locations within facilities (rooms, zones)
    "work_order",  # Maintenance or repair task orders
    "team",  # Maintenance crews or specialized teams
    "budget",  # Financial aspects of maintenance (costs, expenses)
    "tool",  # Tools or instruments used during maintenance
    "training",  # Skills or certifications required
    "priority",  # Urgency levels (critical, high, low)
    "feedback",  # Observations or reports from inspections or workers
    "timeline",  # Timeframes or deadlines for actions
    "outcome",  # Results or resolutions of maintenance activities

    # Expanded Types
    "equipment_system",  # Entire systems (e.g., generator systems, cooling systems)
    "operating_mode",  # Specific states or configurations (e.g., "IN service", "OUT service")
    "test_procedure",  # Specific tests mentioned (e.g., "testing pumps", "testing alternators")
    "control_system",  # Specific control systems (e.g., CSF, excitation systems)
    "inspection_type",  # Specific types of inspections (e.g., "seal test", "water cooling test")
    "subcomponent",  # More granular parts (e.g., "stator", "regulator", "nozzle")
    "maintenance_report",  # Specific reports related to maintenance history or reviews
    "historical_data",  # Logs, reviews, or historical operational data
    "manual",  # Operation and maintenance manuals (specific documents)
    "flow_diagram",  # Process or system diagrams
    "safety_mechanism",  # Specific safety protocols or mechanisms (e.g., "lockout-tagout")
    "energy_source",  # Specific power systems (e.g., "H2", "AIR", "CO2")
    "technical_data",  # Specific technical parameters or datasets
    "process_step",  # Individual steps in operational or maintenance workflows
    "cooling_system",  # Specific cooling system references (e.g., water cooling, gas cooling)
    "generator_component",  # Components of the generator (e.g., "rotor", "exciter")
    "tagging_system",  # Code systems (e.g., "KKS coding system")
    "revision",  # Specific revisions or review cycles (e.g., "2020 revision")
    "emergency_procedure",  # Emergency-specific protocols (e.g., "emergency pump test")
    "task",  # Individual tasks (e.g., resetting, alternator security)
    "data_sheet",  # Documents containing critical technical specifications

    # manually added
    "identifier",
]
tuple_delimiter = "|"  # MSFT GraphRAG uses <|>, but I noticed simpler models mess this delimiter up, so we'll use a simpler one
record_delimiter = "##"
completion_delimiter = "<|COMPLETE|>"
