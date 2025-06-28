"""
Cross-Modal Relationship Mapping Module

This module implements the automated relationship inference algorithms
described in the README for establishing semantic connections between
textual entities and multimodal components.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

from lightrag.utils import logger, compute_mdhash_id
from lightrag.kg.shared_storage import get_namespace_data, get_pipeline_status_lock


@dataclass
class EntityInfo:
    """Entity information for relationship mapping"""
    entity_name: str
    entity_type: str
    description: str
    source_id: str
    file_path: str
    content: str = ""
    chunk_id: str = ""


@dataclass
class RelationshipInfo:
    """Relationship information"""
    source_entity: str
    target_entity: str
    relationship_type: str
    description: str
    weight: float
    source_id: str
    file_path: str


class CrossModalRelationshipManager:
    """Manages cross-modal relationship inference and creation"""
    
    def __init__(self, lightrag, llm_model_func):
        """Initialize the relationship manager"""
        self.lightrag = lightrag
        self.llm_model_func = llm_model_func
        self.logger = logging.getLogger("CrossModalRelationshipManager")
        self.logger.setLevel(logging.INFO)  # Ensure we can see INFO and DEBUG messages
        
        # Add console handler if none exists
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(name)s: %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        self.logger.propagate = True  # Ensure messages propagate to parent loggers
        
        # Use LightRAG's storage instances
        self.text_chunks_db = lightrag.text_chunks
        self.chunks_vdb = lightrag.chunks_vdb
        self.entities_vdb = lightrag.entities_vdb
        self.relationships_vdb = lightrag.relationships_vdb
        self.knowledge_graph_inst = lightrag.chunk_entity_relation_graph
        
        # Relationship type mappings
        self.relationship_types = {
            "image": ["illustrated_by", "contains_image", "visualized_in", "depicted_in"],
            "table": ["explained_by", "contains_table", "summarized_in", "detailed_in"],
            "equation": ["formulated_in", "contains_equation", "mathematically_expressed_in"],
            "generic": ["related_to", "contains_content", "supplements"]
        }

    async def extract_entities_from_processing(self, file_path: str) -> Tuple[List[EntityInfo], List[EntityInfo]]:
        """
        Extract text entities and modal entities from the knowledge graph
        for a specific file
        """
        text_entities = []
        modal_entities = []
        
        try:
            # Get all entity labels from the knowledge graph
            all_labels = await self.knowledge_graph_inst.get_all_labels()
            self.logger.info(f"Found {len(all_labels)} total entity labels in knowledge graph")
            
            # Extract just the filename from the full path for matching
            target_filename = os.path.basename(file_path)
            self.logger.info(f"Looking for entities with filename: {target_filename}")
            
            # Get node data for each label
            for entity_id in all_labels:
                entity_data = await self.knowledge_graph_inst.get_node(entity_id)
                if entity_data:
                    self.logger.debug(f"Entity {entity_id} data: {entity_data}")
                    
                    # Check if this entity belongs to the target file
                    entity_file_path = entity_data.get("file_path", "")
                    entity_filename = os.path.basename(entity_file_path) if entity_file_path else ""
                    
                    if entity_filename == target_filename:
                        entity_info = EntityInfo(
                            entity_name=entity_data.get("entity_id", ""),
                            entity_type=entity_data.get("entity_type", ""),
                            description=entity_data.get("description", ""),
                            source_id=entity_data.get("source_id", ""),
                            file_path=entity_data.get("file_path", ""),
                            content=entity_data.get("content", ""),
                            chunk_id=entity_data.get("source_id", "")
                        )
                        
                        # Categorize entities
                        if entity_info.entity_type in ["image", "table", "equation"]:
                            modal_entities.append(entity_info)
                        else:
                            text_entities.append(entity_info)
                    else:
                        self.logger.debug(f"Entity {entity_id} filename '{entity_filename}' doesn't match target '{target_filename}'")
                else:
                    self.logger.debug(f"No data found for entity {entity_id}")
            
            self.logger.info(f"Extracted {len(text_entities)} text entities and {len(modal_entities)} modal entities for file {target_filename}")
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
        
        return text_entities, modal_entities

    async def infer_cross_modal_relationships(
        self, 
        text_entities: List[EntityInfo], 
        modal_entities: List[EntityInfo],
        document_context: str = ""
    ) -> List[RelationshipInfo]:
        """
        Use LLM to infer relationships between text and modal entities
        """
        relationships = []
        
        if not text_entities or not modal_entities:
            self.logger.info(f"No relationships to infer: text_entities={len(text_entities)}, modal_entities={len(modal_entities)}")
            return relationships
        
        self.logger.info(f"Inferring relationships between {len(text_entities)} text entities and {len(modal_entities)} modal entities")
        
        # Log some sample entities for debugging
        self.logger.info("Sample text entities:")
        for i, entity in enumerate(text_entities[:3]):  # Show first 3
            self.logger.info(f"  {i+1}. {entity.entity_name} ({entity.entity_type}): {entity.description[:100]}...")
        
        self.logger.info("Sample modal entities:")
        for i, entity in enumerate(modal_entities[:3]):  # Show first 3
            self.logger.info(f"  {i+1}. {entity.entity_name} ({entity.entity_type}): {entity.description[:100]}...")
        
        # Create batches for processing
        batch_size = 5  # Process in smaller batches to avoid token limits
        text_batches = [text_entities[i:i + batch_size] for i in range(0, len(text_entities), batch_size)]
        modal_batches = [modal_entities[i:i + batch_size] for i in range(0, len(modal_entities), batch_size)]
        
        self.logger.info(f"Created {len(text_batches)} text batches and {len(modal_batches)} modal batches for processing")
        
        for text_batch_idx, text_batch in enumerate(text_batches):
            for modal_batch_idx, modal_batch in enumerate(modal_batches):
                self.logger.info(f"Processing batch {text_batch_idx+1}/{len(text_batches)} x {modal_batch_idx+1}/{len(modal_batches)}")
                batch_relationships = await self._process_relationship_batch(
                    text_batch, modal_batch, document_context
                )
                relationships.extend(batch_relationships)
                self.logger.info(f"Batch completed with {len(batch_relationships)} relationships")
        
        self.logger.info(f"Inferred {len(relationships)} cross-modal relationships total")
        return relationships

    async def _process_relationship_batch(
        self, 
        text_entities: List[EntityInfo], 
        modal_entities: List[EntityInfo],
        document_context: str
    ) -> List[RelationshipInfo]:
        """Process a batch of entities for relationship inference"""
        relationships = []
        
        self.logger.info(f"Processing batch with {len(text_entities)} text entities and {len(modal_entities)} modal entities")
        
        # Build prompt for relationship inference
        prompt = self._build_relationship_prompt(text_entities, modal_entities, document_context)
        
        try:
            self.logger.info("Calling LLM for relationship inference...")
            response = await self.llm_model_func(
                prompt,
                system_prompt="You are an expert at analyzing relationships between text content and multimodal elements in documents. Provide accurate relationship analysis in JSON format."
            )
            
            self.logger.info(f"LLM response received (length: {len(response)})")
            self.logger.debug(f"LLM response: {response[:500]}...")
            
            # Parse response
            parsed_relationships = self._parse_relationship_response(response, text_entities, modal_entities)
            self.logger.info(f"Parsed {len(parsed_relationships)} relationships from LLM response")
            relationships.extend(parsed_relationships)
            
        except Exception as e:
            self.logger.error(f"Error processing relationship batch: {e}")
            self.logger.debug("Exception details:", exc_info=True)
        
        return relationships

    def _build_relationship_prompt(
        self, 
        text_entities: List[EntityInfo], 
        modal_entities: List[EntityInfo],
        document_context: str
    ) -> str:
        """Build prompt for relationship inference"""
        
        text_entities_str = "\n".join([
            f"- {entity.entity_name} ({entity.entity_type}): {entity.description}"
            for entity in text_entities
        ])
        
        modal_entities_str = "\n".join([
            f"- {entity.entity_name} ({entity.entity_type}): {entity.description}"
            for entity in modal_entities
        ])
        
        prompt = f"""
        Analyze potential relationships between text entities and multimodal elements in this document.

        TEXT ENTITIES:
        {text_entities_str}

        MULTIMODAL ENTITIES:
        {modal_entities_str}

        DOCUMENT CONTEXT:
        {document_context[:1000]}...

        For each potential relationship, determine:
        1. If there's a meaningful connection between a text entity and a modal entity
        2. The type of relationship (e.g., "illustrated_by", "explained_by", "contains_image", "contains_table")
        3. A description of the relationship
        4. A confidence weight (0.1 to 10.0)

        Return your analysis as a JSON array of relationships:
        [
            {{
                "source_entity": "text_entity_name",
                "target_entity": "modal_entity_name", 
                "relationship_type": "relationship_type",
                "description": "description of the relationship",
                "weight": confidence_score
            }}
        ]

        Only include relationships where there's a clear semantic connection. If no relationships exist, return an empty array [].
        """
        
        return prompt

    def _parse_relationship_response(
        self, 
        response: str, 
        text_entities: List[EntityInfo], 
        modal_entities: List[EntityInfo]
    ) -> List[RelationshipInfo]:
        """Parse LLM response to extract relationships"""
        relationships = []
        
        try:
            self.logger.info("Parsing LLM response for relationships...")
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                self.logger.warning("No JSON array found in LLM response")
                self.logger.debug(f"Response content: {response[:200]}...")
                return relationships
            
            json_str = json_match.group(0)
            self.logger.info(f"Found JSON array: {json_str[:200]}...")
            
            relationship_data = json.loads(json_str)
            self.logger.info(f"Parsed {len(relationship_data)} relationship entries from JSON")
            
            for i, rel_data in enumerate(relationship_data):
                self.logger.debug(f"Processing relationship {i+1}: {rel_data}")
                
                # Validate relationship data
                if not all(key in rel_data for key in ["source_entity", "target_entity", "relationship_type", "description", "weight"]):
                    self.logger.warning(f"Relationship {i+1} missing required fields: {rel_data}")
                    continue
                
                # Find corresponding entities
                source_entity = next((e for e in text_entities if e.entity_name == rel_data["source_entity"]), None)
                target_entity = next((e for e in modal_entities if e.entity_name == rel_data["target_entity"]), None)
                
                if source_entity and target_entity:
                    self.logger.info(f"Found matching entities: {source_entity.entity_name} -> {target_entity.entity_name}")
                    relationship = RelationshipInfo(
                        source_entity=source_entity.entity_name,
                        target_entity=target_entity.entity_name,
                        relationship_type=rel_data["relationship_type"],
                        description=rel_data["description"],
                        weight=float(rel_data["weight"]),
                        source_id=source_entity.source_id,
                        file_path=source_entity.file_path
                    )
                    relationships.append(relationship)
                else:
                    self.logger.warning(f"Could not find matching entities for relationship {i+1}:")
                    if not source_entity:
                        self.logger.warning(f"  Source entity '{rel_data['source_entity']}' not found in text entities")
                    if not target_entity:
                        self.logger.warning(f"  Target entity '{rel_data['target_entity']}' not found in modal entities")
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.error(f"Error parsing relationship response: {e}")
            self.logger.debug("Exception details:", exc_info=True)
        
        return relationships

    async def create_relationships_in_graph(self, relationships: List[RelationshipInfo]) -> None:
        """Create the inferred relationships in the knowledge graph"""
        
        self.logger.info(f"Creating {len(relationships)} cross-modal relationships in knowledge graph")
        
        for relationship in relationships:
            try:
                # Create relationship in knowledge graph
                relation_data = {
                    "description": relationship.description,
                    "keywords": relationship.relationship_type,
                    "source_id": relationship.source_id,
                    "weight": relationship.weight,
                    "file_path": relationship.file_path,
                }
                
                await self.knowledge_graph_inst.upsert_edge(
                    relationship.source_entity, 
                    relationship.target_entity, 
                    relation_data
                )
                
                # Create relationship in vector database
                relation_id = compute_mdhash_id(
                    relationship.source_entity + relationship.target_entity, 
                    prefix="rel-"
                )
                
                relation_vdb_data = {
                    relation_id: {
                        "src_id": relationship.source_entity,
                        "tgt_id": relationship.target_entity,
                        "keywords": relationship.relationship_type,
                        "content": f"{relationship.relationship_type}\t{relationship.source_entity}\n{relationship.target_entity}\n{relationship.description}",
                        "source_id": relationship.source_id,
                        "file_path": relationship.file_path,
                    }
                }
                
                await self.relationships_vdb.upsert(relation_vdb_data)
                
            except Exception as e:
                self.logger.error(f"Error creating relationship {relationship.source_entity} -> {relationship.target_entity}: {e}")
        
        # Ensure all storage updates are complete
        await self._insert_done()

    async def _insert_done(self) -> None:
        """Ensure all storage updates are complete"""
        await asyncio.gather(
            *[
                storage_inst.index_done_callback()
                for storage_inst in [
                    self.text_chunks_db,
                    self.chunks_vdb,
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.knowledge_graph_inst,
                ]
            ]
        )

    async def process_document_relationships(self, file_path: str, document_context: str = "") -> Dict[str, Any]:
        """
        Complete workflow for processing cross-modal relationships for a document
        """
        self.logger.info(f"Processing cross-modal relationships for {file_path}")
        
        try:
            # Extract entities
            text_entities, modal_entities = await self.extract_entities_from_processing(file_path)
            
            if not text_entities or not modal_entities:
                self.logger.info("No text entities or modal entities found for relationship processing")
                return {"relationships_created": 0, "text_entities": len(text_entities), "modal_entities": len(modal_entities)}
            
            # Infer relationships
            relationships = await self.infer_cross_modal_relationships(
                text_entities, modal_entities, document_context
            )
            
            # Create relationships in graph
            await self.create_relationships_in_graph(relationships)
            
            result = {
                "relationships_created": len(relationships),
                "text_entities": len(text_entities),
                "modal_entities": len(modal_entities),
                "relationships": [
                    {
                        "source": rel.source_entity,
                        "target": rel.target_entity,
                        "type": rel.relationship_type,
                        "description": rel.description,
                        "weight": rel.weight
                    }
                    for rel in relationships
                ]
            }
            
            self.logger.info(f"Successfully created {len(relationships)} cross-modal relationships")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing document relationships: {e}")
            return {"error": str(e), "relationships_created": 0} 