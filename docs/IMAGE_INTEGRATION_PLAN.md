# Image Integration Plan for CRM_RAG

**Created**: 2026-01-26
**Status**: Planning
**Author**: Claude Code Analysis

This document provides a comprehensive plan for integrating image retrieval into the CRM_RAG chatbot system. The goal is to display images that are linked to entities in the RDF knowledge graph, without generating or correlating unlinked images.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Image Data Patterns in RDF](#image-data-patterns-in-rdf)
4. [Per-Dataset Image Schema Configuration](#per-dataset-image-schema-configuration)
5. [Architecture Overview](#architecture-overview)
6. [Implementation Phases](#implementation-phases)
   - [Phase 1: Configuration Schema](#phase-1-configuration-schema)
   - [Phase 2: Image Resolver Module](#phase-2-image-resolver-module)
   - [Phase 3: IIIF Support](#phase-3-iiif-support)
   - [Phase 4: Backend Integration](#phase-4-backend-integration)
   - [Phase 5: Frontend Display](#phase-5-frontend-display)
   - [Phase 6: Document Generation Enhancement](#phase-6-document-generation-enhancement)
7. [API Specifications](#api-specifications)
8. [Security Considerations](#security-considerations)
9. [Testing Strategy](#testing-strategy)
10. [Migration Guide](#migration-guide)

---

## Executive Summary

### Goal
Enable the CRM_RAG chatbot to display images associated with entities in the knowledge graph, enhancing answers with visual context when available.

### Key Principles
1. **No image generation**: Only display images that exist in the RDF graph
2. **No unlinked correlation**: Images must have explicit RDF relationships to entities
3. **Per-dataset configuration**: Each dataset defines its own image schema and predicates
4. **Graceful degradation**: Handle missing images, CORS issues, failed fetches silently
5. **Performance-first**: Cache lookups, use thumbnails, lazy-load when possible

### Scope
- Extract image URLs from RDF graph using configurable predicates
- Support IIIF manifests with thumbnail generation
- Support direct image URLs (Wikimedia Commons, institutional servers)
- Display images in chat sources section
- Enable/disable per dataset

---

## Current State Analysis

### Existing Image-Related Code

| Location | Current Behavior | Notes |
|----------|------------------|-------|
| `universal_rag_system.py:2709` | Extracts Wikidata P18 (image) property | Works but not displayed |
| `static/js/chat.js:425-427` | Explicitly skips displaying image URLs | `if (propName === 'image') { continue; }` |
| `graph_document_store.py` | `metadata` dict has no image fields | Needs extension |
| `bulk_generate_documents.py` | Does not extract image URLs | Processes literals only |

### GraphDocument Metadata Structure (Current)

```python
metadata = {
    "label": "Entity Label",
    "uri": "http://example.org/entity",
    "type": "Entity Type",
    "raw_triples": [...],
    "doc_id": "unique_id"
}
```

### API Response Structure (Current)

```json
{
  "answer": "...",
  "sources": [
    {
      "id": 0,
      "entity_uri": "...",
      "entity_label": "...",
      "type": "graph",
      "entity_type": "...",
      "raw_triples": [...]
    }
  ]
}
```

---

## Image Data Patterns in RDF

Based on analysis of the MAH (Museum) dataset, the following patterns were identified:

### Pattern 1: IIIF Manifests via Digital Information Objects

```
Artwork (E22_Man-Made_Object)
    └── crm:P129i_is_subject_of / crm:carries → Digital (E73_Information_Object)
            └── rdf:value / crm:P1_is_identified_by → IIIF Manifest URL
```

**Example from data**:
```
URI: https://data.mahmah.ch/archive/item/2199736/digital
Type: Information Object
Value: https://iiif.hedera.unige.ch/iiif/manifests/mah/30097893
Relationships:
  - digital is identified by https://iiif.hedera.unige.ch/iiif/manifests/mah/30097893
  - Artistes et décorateurs carries digital
```

**SPARQL Pattern**:
```sparql
SELECT ?manifestUrl WHERE {
    <entity_uri> (crm:carries|crm:P129i_is_subject_of|^crm:P67_refers_to) ?digital .
    ?digital (rdf:value|crm:P1_is_identified_by) ?manifestUrl .
    FILTER(CONTAINS(STR(?manifestUrl), "iiif"))
}
```

### Pattern 2: Direct Image URLs via rdf:value

```
Image Identifier Entity
    └── rdf:value → Direct Image URL (Wikimedia Commons, etc.)
```

**Example from data**:
```
URI: https://data.mahmah.ch/image/agent/57883/url
Type: Identifier
Value: http://commons.wikimedia.org/wiki/Special:FilePath/Dahl,_Michael_-_Queen_Anne_-_NPG_6187.jpg
Relationships:
  - 57883 is identified by url
```

**SPARQL Pattern**:
```sparql
SELECT ?imageUrl WHERE {
    <entity_uri> crm:P1_is_identified_by ?identifier .
    ?identifier rdf:value ?imageUrl .
    FILTER(
        CONTAINS(STR(?imageUrl), ".jpg") ||
        CONTAINS(STR(?imageUrl), ".png") ||
        CONTAINS(STR(?imageUrl), ".jpeg") ||
        CONTAINS(STR(?imageUrl), ".tiff") ||
        CONTAINS(STR(?imageUrl), "wikimedia") ||
        CONTAINS(STR(?imageUrl), "Special:FilePath")
    )
}
```

### Pattern 3: Has Representation Relationship

```
Artwork (E22_Man-Made_Object)
    └── crm:P138_has_representation → Visual Item ID
```

**Example from data**:
```
- Kilgaren Castle has representation 327484
- La Vierge à l'Enfant... has representation 1415339
```

**SPARQL Pattern**:
```sparql
SELECT ?representationId WHERE {
    <entity_uri> crm:P138_has_representation ?representation .
    BIND(STRAFTER(STR(?representation), "/") AS ?representationId)
}
```

### Pattern 4: Visual Item Type

Some entities are explicitly typed as Visual Items:
```
Type: Visual Item (E36_Visual_Item)
```

### Pattern 5: Wikidata P18 (External)

Already implemented but not displayed:
```python
property_map = {
    "P18": "image",  # Wikidata image property
}
```

### Pattern 6: foaf:depiction (Standard Linked Data)

Common in linked data but needs verification per dataset:
```sparql
SELECT ?imageUrl WHERE {
    <entity_uri> foaf:depiction ?imageUrl .
}
```

---

## Per-Dataset Image Schema Configuration

Each dataset can have different predicates and patterns for linking images. This requires a flexible, per-dataset configuration.

### Configuration Schema

**File**: `config/datasets.yaml`

```yaml
datasets:
  mah:
    name: mah
    display_name: "Museum Collection"
    description: "Museum artworks, artists, and exhibitions"
    endpoint: "http://localhost:3030/MAH/sparql"

    # Image configuration for this dataset
    images:
      enabled: true

      # IIIF Configuration
      iiif:
        enabled: true
        # Base URL for IIIF Image API (used to construct thumbnail URLs)
        image_api_base: "https://iiif.hedera.unige.ch/iiif/2"
        # Default thumbnail size (IIIF size parameter)
        thumbnail_size: "200,"
        # Thumbnail quality
        thumbnail_quality: "default"
        # Thumbnail format
        thumbnail_format: "jpg"
        # Manifest fetch timeout (seconds)
        manifest_timeout: 10
        # Cache manifest responses (seconds, 0 = no cache)
        manifest_cache_ttl: 3600

      # Predicates to search for image URLs (in order of priority)
      # Each predicate can be a URI or a SPARQL property path
      predicates:
        # Pattern 1: Digital objects with IIIF manifests
        - path: "(crm:carries|crm:P129i_is_subject_of|^crm:P67_refers_to)/(rdf:value|crm:P1_is_identified_by)"
          type: "iiif_manifest"
          filter: "CONTAINS(STR(?value), 'iiif')"
          description: "IIIF manifests via digital information objects"

        # Pattern 2: Direct image URLs via identifier
        - path: "crm:P1_is_identified_by/rdf:value"
          type: "direct"
          filter: "REGEX(STR(?value), '\\.(jpg|jpeg|png|tiff|gif)$', 'i') || CONTAINS(STR(?value), 'wikimedia')"
          description: "Direct image URLs via identifiers"

        # Pattern 3: Has representation (needs secondary lookup)
        - path: "crm:P138_has_representation"
          type: "representation_id"
          description: "Visual representation IDs (requires secondary lookup)"
          # Template for constructing image URL from representation ID
          url_template: "https://iiif.hedera.unige.ch/iiif/manifests/mah/{id}"

        # Pattern 4: Standard foaf:depiction
        - path: "foaf:depiction"
          type: "direct"
          description: "Standard linked data image predicate"

      # URL patterns to recognize and handle
      url_handlers:
        # Wikimedia Commons special handling
        - pattern: "commons.wikimedia.org/wiki/Special:FilePath/"
          type: "wikimedia_commons"
          # Transform to thumbnail URL
          thumbnail_transform: "https://commons.wikimedia.org/wiki/Special:FilePath/{filename}?width=200"

        # Direct IIIF Image API URLs
        - pattern: "/iiif/2/"
          type: "iiif_image"
          thumbnail_transform: "{base}/full/200,/0/default.jpg"

        # Generic image URLs
        - pattern: "\\.(jpg|jpeg|png|gif|tiff)$"
          type: "direct"
          # No transform, use as-is (or proxy for CORS)

      # Display settings
      display:
        max_images_per_entity: 3
        show_in_answer: false  # Show images inline in answer text
        show_in_sources: true  # Show images in sources section
        thumbnail_width: 150
        thumbnail_height: 100
        lightbox_enabled: true
        lazy_load: true

    interface:
      # ... existing interface config ...

  asinou:
    name: asinou
    display_name: "Asinou Church"
    description: "Asinou church dataset with frescoes, iconography"
    endpoint: "http://localhost:3030/asinou/sparql"

    # Different image configuration for Asinou dataset
    images:
      enabled: true

      iiif:
        enabled: false  # Asinou might not use IIIF

      predicates:
        # Asinou-specific predicates (example - needs verification)
        - path: "crm:P138_represents"
          type: "direct"
          description: "Visual representation"

        - path: "vir:K17_has_visual_prototype"
          type: "direct"
          description: "VIR visual prototype"

      display:
        max_images_per_entity: 5
        show_in_sources: true
        thumbnail_width: 200
```

### Predicate Configuration Schema

```yaml
# Each predicate entry has the following structure:
predicate:
  # SPARQL property path (can include inverse ^, alternatives |, sequences /)
  path: "crm:P138_has_representation"

  # Type of value expected:
  # - "direct": URL points directly to an image file
  # - "iiif_manifest": URL points to a IIIF manifest JSON
  # - "iiif_image": URL is a IIIF Image API URL
  # - "representation_id": Value is an ID that needs template expansion
  # - "wikimedia_commons": Wikimedia Commons file path
  type: "direct"

  # Optional SPARQL FILTER clause (variable is always ?value)
  filter: "CONTAINS(STR(?value), '.jpg')"

  # Human-readable description
  description: "Direct image URLs"

  # For type="representation_id": template for constructing URL
  # {id} will be replaced with the extracted ID
  url_template: "https://example.org/images/{id}.jpg"

  # Priority (lower = higher priority, default = 100)
  priority: 10
```

### Namespace Prefixes

The configuration should support custom namespace prefixes per dataset:

```yaml
datasets:
  mah:
    # Custom namespace prefixes for this dataset
    namespaces:
      crm: "http://www.cidoc-crm.org/cidoc-crm/"
      crmdig: "http://www.ics.forth.gr/isl/CRMdig/"
      vir: "http://www.ics.forth.gr/isl/VIR/"
      foaf: "http://xmlns.com/foaf/0.1/"
      rdf: "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
      aat: "http://vocab.getty.edu/aat/"
      # Dataset-specific namespaces
      mah: "https://data.mahmah.ch/"
```

---

## Architecture Overview

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Flask App (main.py)                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │  /api/chat      │  │ /api/entity/    │  │ /api/datasets/{id}/    │  │
│  │                 │  │   {uri}/images  │  │   images/config        │  │
│  └────────┬────────┘  └────────┬────────┘  └────────────┬───────────┘  │
└───────────┼─────────────────────┼───────────────────────┼──────────────┘
            │                     │                       │
            ▼                     ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        UniversalRagSystem                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     answer_question()                            │   │
│  │  - Retrieve documents                                            │   │
│  │  - Get images for retrieved entities (NEW)                       │   │
│  │  - Build context with image references                           │   │
│  │  - Generate answer                                               │   │
│  │  - Return sources with images                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ImageResolver (NEW)                           │   │
│  │  - Load dataset image config                                     │   │
│  │  - Execute SPARQL queries for image URLs                         │   │
│  │  - Resolve URL types (IIIF, direct, wikimedia)                   │   │
│  │  - Generate thumbnail URLs                                       │   │
│  │  - Cache results                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    IIIFClient (NEW)                              │   │
│  │  - Fetch and parse IIIF manifests                                │   │
│  │  - Extract image service URLs                                    │   │
│  │  - Construct thumbnail URLs                                      │   │
│  │  - Handle IIIF Image API                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         SPARQL Endpoint                                  │
│                    (Fuseki / GraphDB / etc.)                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow for Image Retrieval

```
1. User asks question
        │
        ▼
2. RAG retrieves relevant entities
        │
        ▼
3. For each retrieved entity:
   ┌────────────────────────────────────────────┐
   │  a. Check if images enabled for dataset    │
   │  b. Load image predicates config           │
   │  c. Execute SPARQL query with predicates   │
   │  d. For each found URL:                    │
   │     - Detect URL type                      │
   │     - If IIIF manifest: fetch & parse      │
   │     - Generate thumbnail URL               │
   │     - Add to entity's image list           │
   └────────────────────────────────────────────┘
        │
        ▼
4. Build response with images in sources
        │
        ▼
5. Frontend renders images in sources section
```

---

## Implementation Phases

### Phase 1: Configuration Schema

**Goal**: Define and load per-dataset image configuration

**Files to create/modify**:
- `config/datasets.yaml` - Add image schema
- `dataset_manager.py` - Load and validate image config
- `config/image_schema.py` (NEW) - Pydantic models for validation

#### 1.1 Create Image Configuration Schema

**New file**: `config/image_schema.py`

```python
"""
Image configuration schema for CRM_RAG datasets.
Uses Pydantic for validation and type safety.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class ImageUrlType(str, Enum):
    """Types of image URL sources"""
    DIRECT = "direct"
    IIIF_MANIFEST = "iiif_manifest"
    IIIF_IMAGE = "iiif_image"
    REPRESENTATION_ID = "representation_id"
    WIKIMEDIA_COMMONS = "wikimedia_commons"


class PredicateConfig(BaseModel):
    """Configuration for a single image predicate"""
    path: str = Field(..., description="SPARQL property path")
    type: ImageUrlType = Field(..., description="Type of value expected")
    filter: Optional[str] = Field(None, description="SPARQL FILTER clause")
    description: Optional[str] = Field(None, description="Human-readable description")
    url_template: Optional[str] = Field(None, description="URL template for representation_id type")
    priority: int = Field(100, description="Priority (lower = higher)")

    @validator('url_template')
    def validate_template(cls, v, values):
        if values.get('type') == ImageUrlType.REPRESENTATION_ID and not v:
            raise ValueError("url_template required for representation_id type")
        return v


class UrlHandlerConfig(BaseModel):
    """Configuration for URL pattern handling"""
    pattern: str = Field(..., description="Regex pattern to match URLs")
    type: ImageUrlType = Field(..., description="Handler type")
    thumbnail_transform: Optional[str] = Field(None, description="Transform template for thumbnails")


class IIIFConfig(BaseModel):
    """IIIF-specific configuration"""
    enabled: bool = Field(True, description="Enable IIIF support")
    image_api_base: Optional[str] = Field(None, description="Base URL for IIIF Image API")
    thumbnail_size: str = Field("200,", description="IIIF size parameter")
    thumbnail_quality: str = Field("default", description="IIIF quality parameter")
    thumbnail_format: str = Field("jpg", description="IIIF format parameter")
    manifest_timeout: int = Field(10, description="Manifest fetch timeout in seconds")
    manifest_cache_ttl: int = Field(3600, description="Cache TTL in seconds (0 = no cache)")


class DisplayConfig(BaseModel):
    """Image display configuration"""
    max_images_per_entity: int = Field(3, description="Maximum images to show per entity")
    show_in_answer: bool = Field(False, description="Show images inline in answer")
    show_in_sources: bool = Field(True, description="Show images in sources section")
    thumbnail_width: int = Field(150, description="Thumbnail width in pixels")
    thumbnail_height: Optional[int] = Field(None, description="Thumbnail height (None = auto)")
    lightbox_enabled: bool = Field(True, description="Enable lightbox for full-size view")
    lazy_load: bool = Field(True, description="Lazy load images")


class DatasetImageConfig(BaseModel):
    """Complete image configuration for a dataset"""
    enabled: bool = Field(True, description="Enable image support for this dataset")
    iiif: IIIFConfig = Field(default_factory=IIIFConfig, description="IIIF configuration")
    predicates: List[PredicateConfig] = Field(default_factory=list, description="Image predicates")
    url_handlers: List[UrlHandlerConfig] = Field(default_factory=list, description="URL handlers")
    display: DisplayConfig = Field(default_factory=DisplayConfig, description="Display settings")
    namespaces: Dict[str, str] = Field(default_factory=dict, description="Custom namespace prefixes")

    def get_sorted_predicates(self) -> List[PredicateConfig]:
        """Return predicates sorted by priority"""
        return sorted(self.predicates, key=lambda p: p.priority)


def load_image_config(dataset_config: dict) -> Optional[DatasetImageConfig]:
    """
    Load and validate image configuration from dataset config.

    Args:
        dataset_config: Raw dataset configuration dictionary

    Returns:
        DatasetImageConfig if images config exists, None otherwise
    """
    if 'images' not in dataset_config:
        return None

    try:
        return DatasetImageConfig(**dataset_config['images'])
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            f"Invalid image configuration: {e}. Images disabled for this dataset."
        )
        return None
```

#### 1.2 Modify Dataset Manager

**File**: `dataset_manager.py`

Add image config loading:

```python
# Add import at top
from config.image_schema import DatasetImageConfig, load_image_config

class DatasetManager:
    def __init__(self, datasets_config, llm_config):
        # ... existing code ...
        self.image_configs = {}  # Cache for image configs

    def get_image_config(self, dataset_id: str) -> Optional[DatasetImageConfig]:
        """Get image configuration for a dataset"""
        if dataset_id not in self.image_configs:
            dataset = self.datasets.get(dataset_id)
            if dataset:
                self.image_configs[dataset_id] = load_image_config(dataset)
            else:
                self.image_configs[dataset_id] = None
        return self.image_configs[dataset_id]

    def images_enabled(self, dataset_id: str) -> bool:
        """Check if images are enabled for a dataset"""
        config = self.get_image_config(dataset_id)
        return config is not None and config.enabled
```

#### 1.3 Add Default Namespace Prefixes

**File**: `config/namespaces.py` (NEW)

```python
"""
Default namespace prefixes for SPARQL queries.
Can be overridden per-dataset in datasets.yaml.
"""

DEFAULT_NAMESPACES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "crm": "http://www.cidoc-crm.org/cidoc-crm/",
    "crmdig": "http://www.ics.forth.gr/isl/CRMdig/",
    "vir": "http://www.ics.forth.gr/isl/VIR/",
    "frbroo": "http://iflastandards.info/ns/fr/frbr/frbroo/",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "schema": "http://schema.org/",
    "aat": "http://vocab.getty.edu/aat/",
    "wd": "http://www.wikidata.org/entity/",
    "wdt": "http://www.wikidata.org/prop/direct/",
}


def get_prefix_declarations(custom_namespaces: dict = None) -> str:
    """
    Generate SPARQL PREFIX declarations.

    Args:
        custom_namespaces: Optional dict of custom prefixes to merge/override

    Returns:
        String of PREFIX declarations for SPARQL query
    """
    namespaces = DEFAULT_NAMESPACES.copy()
    if custom_namespaces:
        namespaces.update(custom_namespaces)

    return "\n".join(
        f"PREFIX {prefix}: <{uri}>"
        for prefix, uri in sorted(namespaces.items())
    )
```

---

### Phase 2: Image Resolver Module

**Goal**: Create the core module for resolving image URLs from RDF data

**New file**: `image_resolver.py`

```python
"""
Image URL resolver for CRM_RAG.
Extracts and resolves image URLs from RDF graph based on dataset configuration.
"""

import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse, quote
from functools import lru_cache
import time

from SPARQLWrapper import SPARQLWrapper, JSON

from config.image_schema import (
    DatasetImageConfig,
    PredicateConfig,
    ImageUrlType,
    UrlHandlerConfig
)
from config.namespaces import get_prefix_declarations

logger = logging.getLogger(__name__)


@dataclass
class ResolvedImage:
    """Represents a resolved image with URLs and metadata"""
    original_url: str
    thumbnail_url: Optional[str] = None
    full_url: Optional[str] = None
    url_type: ImageUrlType = ImageUrlType.DIRECT
    source: str = "graph"  # "graph" or "wikidata"
    predicate: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    label: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "url": self.full_url or self.original_url,
            "thumbnail_url": self.thumbnail_url or self.original_url,
            "type": self.url_type.value,
            "source": self.source,
            "predicate": self.predicate,
            "label": self.label,
            "error": self.error
        }


class ImageResolver:
    """
    Resolves image URLs for entities based on dataset configuration.
    """

    def __init__(self, sparql_endpoint: str, image_config: DatasetImageConfig):
        """
        Initialize the image resolver.

        Args:
            sparql_endpoint: SPARQL endpoint URL
            image_config: Dataset image configuration
        """
        self.endpoint = sparql_endpoint
        self.config = image_config
        self.sparql = SPARQLWrapper(sparql_endpoint)
        self.sparql.setReturnFormat(JSON)

        # Build prefix declarations
        self.prefixes = get_prefix_declarations(image_config.namespaces)

        # Compile URL handler patterns
        self._compiled_handlers = [
            (re.compile(h.pattern, re.IGNORECASE), h)
            for h in image_config.url_handlers
        ]

        # Simple in-memory cache for resolved images
        self._cache: Dict[str, Tuple[List[ResolvedImage], float]] = {}
        self._cache_ttl = 300  # 5 minutes default

        logger.info(f"ImageResolver initialized for endpoint: {sparql_endpoint}")
        logger.info(f"Configured {len(image_config.predicates)} image predicates")

    def get_images_for_entity(
        self,
        entity_uri: str,
        max_images: Optional[int] = None
    ) -> List[ResolvedImage]:
        """
        Get images for an entity.

        Args:
            entity_uri: URI of the entity
            max_images: Maximum number of images to return (None = use config)

        Returns:
            List of ResolvedImage objects
        """
        if not self.config.enabled:
            return []

        # Check cache
        cache_key = f"entity:{entity_uri}"
        if cache_key in self._cache:
            cached_images, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                logger.debug(f"Cache hit for {entity_uri}")
                return cached_images[:max_images] if max_images else cached_images

        max_images = max_images or self.config.display.max_images_per_entity
        images = []

        # Query each predicate in priority order
        for predicate_config in self.config.get_sorted_predicates():
            if len(images) >= max_images:
                break

            try:
                found_images = self._query_predicate(entity_uri, predicate_config)
                for img in found_images:
                    if len(images) >= max_images:
                        break
                    # Avoid duplicates
                    if not any(i.original_url == img.original_url for i in images):
                        images.append(img)
            except Exception as e:
                logger.warning(f"Error querying predicate {predicate_config.path}: {e}")

        # Resolve thumbnails for all images
        for img in images:
            self._resolve_thumbnail(img)

        # Cache results
        self._cache[cache_key] = (images, time.time())

        logger.info(f"Found {len(images)} images for entity {entity_uri}")
        return images

    def _query_predicate(
        self,
        entity_uri: str,
        predicate: PredicateConfig
    ) -> List[ResolvedImage]:
        """
        Query a single predicate for image URLs.

        Args:
            entity_uri: URI of the entity
            predicate: Predicate configuration

        Returns:
            List of ResolvedImage objects
        """
        # Build SPARQL query
        filter_clause = f"FILTER({predicate.filter})" if predicate.filter else ""

        query = f"""
        {self.prefixes}

        SELECT DISTINCT ?value WHERE {{
            <{entity_uri}> {predicate.path} ?value .
            {filter_clause}
        }}
        LIMIT 10
        """

        logger.debug(f"Executing image query for {entity_uri} with path {predicate.path}")

        self.sparql.setQuery(query)
        results = self.sparql.query().convert()

        images = []
        for binding in results.get("results", {}).get("bindings", []):
            value = binding.get("value", {}).get("value")
            if not value:
                continue

            # Handle different predicate types
            if predicate.type == ImageUrlType.REPRESENTATION_ID:
                # Use template to construct URL
                if predicate.url_template:
                    # Extract ID from value
                    id_value = value.split("/")[-1] if "/" in value else value
                    url = predicate.url_template.replace("{id}", id_value)
                    images.append(ResolvedImage(
                        original_url=url,
                        url_type=ImageUrlType.IIIF_MANIFEST,  # Assume IIIF for templates
                        predicate=predicate.path
                    ))
            else:
                images.append(ResolvedImage(
                    original_url=value,
                    url_type=predicate.type,
                    predicate=predicate.path
                ))

        return images

    def _resolve_thumbnail(self, image: ResolvedImage) -> None:
        """
        Resolve thumbnail URL for an image based on its type.

        Args:
            image: ResolvedImage to update with thumbnail URL
        """
        url = image.original_url

        # Try URL handlers first
        for pattern, handler in self._compiled_handlers:
            if pattern.search(url):
                image.url_type = handler.type
                if handler.thumbnail_transform:
                    image.thumbnail_url = self._apply_transform(
                        url, handler.thumbnail_transform
                    )
                    image.full_url = url
                    return

        # Handle by type
        if image.url_type == ImageUrlType.IIIF_MANIFEST:
            # Will be resolved by IIIFClient
            pass
        elif image.url_type == ImageUrlType.IIIF_IMAGE:
            image.thumbnail_url = self._make_iiif_thumbnail(url)
            image.full_url = self._make_iiif_full(url)
        elif image.url_type == ImageUrlType.WIKIMEDIA_COMMONS:
            image.thumbnail_url = self._make_wikimedia_thumbnail(url)
            image.full_url = url
        else:
            # Direct URL - use as-is
            image.thumbnail_url = url
            image.full_url = url

    def _apply_transform(self, url: str, template: str) -> str:
        """Apply a URL transform template"""
        # Extract filename for wikimedia
        if "Special:FilePath" in url:
            filename = url.split("Special:FilePath/")[-1]
            return template.replace("{filename}", quote(filename))

        # Extract base URL for IIIF
        if "/iiif/" in url:
            # Find the base before size/region parameters
            parts = url.split("/")
            base_idx = parts.index("iiif") if "iiif" in parts else -1
            if base_idx >= 0:
                base = "/".join(parts[:base_idx+3])  # Up to image ID
                return template.replace("{base}", base)

        return url  # No transform applied

    def _make_iiif_thumbnail(self, url: str) -> str:
        """Generate IIIF thumbnail URL"""
        config = self.config.iiif
        # Parse IIIF URL and reconstruct with thumbnail parameters
        # IIIF URL format: {scheme}://{server}/{prefix}/{identifier}/{region}/{size}/{rotation}/{quality}.{format}

        # Simple approach: replace size parameter
        parts = url.split("/")
        if len(parts) >= 4:
            # Try to find and replace size parameter (usually "full" or "max" or dimensions)
            for i, part in enumerate(parts):
                if part in ("full", "max") or re.match(r"^\d+,\d*$|^\d*,\d+$|^!\d+,\d+$", part):
                    parts[i] = config.thumbnail_size
                    break

        return "/".join(parts)

    def _make_iiif_full(self, url: str) -> str:
        """Generate IIIF full-size URL"""
        # Replace size with max
        parts = url.split("/")
        for i, part in enumerate(parts):
            if re.match(r"^\d+,\d*$|^\d*,\d+$|^!\d+,\d+$", part):
                parts[i] = "max"
                break
        return "/".join(parts)

    def _make_wikimedia_thumbnail(self, url: str, width: int = 200) -> str:
        """Generate Wikimedia Commons thumbnail URL"""
        # Wikimedia thumbnail URL format:
        # https://upload.wikimedia.org/wikipedia/commons/thumb/{hash}/{filename}/{width}px-{filename}

        if "Special:FilePath" in url:
            filename = url.split("Special:FilePath/")[-1]
            return f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(filename)}?width={width}"

        return url

    def get_images_for_entities(
        self,
        entity_uris: List[str],
        max_images_per_entity: Optional[int] = None
    ) -> Dict[str, List[ResolvedImage]]:
        """
        Get images for multiple entities.

        Args:
            entity_uris: List of entity URIs
            max_images_per_entity: Max images per entity (None = use config)

        Returns:
            Dict mapping entity URI to list of ResolvedImage
        """
        result = {}
        for uri in entity_uris:
            result[uri] = self.get_images_for_entity(uri, max_images_per_entity)
        return result

    def clear_cache(self) -> None:
        """Clear the image cache"""
        self._cache.clear()
        logger.info("Image cache cleared")
```

---

### Phase 3: IIIF Support

**Goal**: Implement IIIF manifest parsing and image service handling

**New file**: `iiif_client.py`

```python
"""
IIIF (International Image Interoperability Framework) client for CRM_RAG.
Handles IIIF Presentation API manifests and Image API URLs.
"""

import logging
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import requests
from functools import lru_cache
import time
import threading

logger = logging.getLogger(__name__)


@dataclass
class IIIFImageInfo:
    """Information about an image from a IIIF manifest"""
    image_id: str
    service_url: str
    width: Optional[int] = None
    height: Optional[int] = None
    format: str = "jpg"
    label: Optional[str] = None

    def get_thumbnail_url(self, size: str = "200,", quality: str = "default") -> str:
        """
        Generate thumbnail URL using IIIF Image API.

        Args:
            size: IIIF size parameter (e.g., "200,", ",200", "200,200", "!200,200")
            quality: IIIF quality parameter (default, color, gray, bitonal)

        Returns:
            Full IIIF Image API URL for thumbnail
        """
        # IIIF Image API URL format:
        # {scheme}://{server}{/prefix}/{identifier}/{region}/{size}/{rotation}/{quality}.{format}
        base = self.service_url.rstrip("/")
        return f"{base}/full/{size}/0/{quality}.{self.format}"

    def get_full_url(self, quality: str = "default") -> str:
        """Generate full-size image URL"""
        base = self.service_url.rstrip("/")
        return f"{base}/full/max/0/{quality}.{self.format}"

    def get_region_url(
        self,
        region: str = "full",
        size: str = "max",
        rotation: int = 0,
        quality: str = "default"
    ) -> str:
        """
        Generate URL with custom IIIF parameters.

        Args:
            region: Region parameter (full, square, x,y,w,h, pct:x,y,w,h)
            size: Size parameter (max, w,, ,h, w,h, !w,h, pct:n)
            rotation: Rotation in degrees (0, 90, 180, 270, or arbitrary with !)
            quality: Quality parameter

        Returns:
            Full IIIF Image API URL
        """
        base = self.service_url.rstrip("/")
        return f"{base}/{region}/{size}/{rotation}/{quality}.{self.format}"


class IIIFClient:
    """
    Client for interacting with IIIF services.

    Supports:
    - IIIF Presentation API 2.1 and 3.0 manifests
    - IIIF Image API 2.1 and 3.0
    """

    def __init__(
        self,
        timeout: int = 10,
        cache_ttl: int = 3600,
        max_cache_size: int = 1000
    ):
        """
        Initialize IIIF client.

        Args:
            timeout: Request timeout in seconds
            cache_ttl: Cache time-to-live in seconds
            max_cache_size: Maximum number of cached manifests
        """
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size

        # Thread-safe cache
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_lock = threading.Lock()

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/ld+json, application/json",
            "User-Agent": "CRM_RAG/1.0 (IIIF Client)"
        })

    def get_images_from_manifest(
        self,
        manifest_url: str
    ) -> List[IIIFImageInfo]:
        """
        Extract image information from a IIIF manifest.

        Args:
            manifest_url: URL of the IIIF manifest

        Returns:
            List of IIIFImageInfo objects
        """
        manifest = self._fetch_manifest(manifest_url)
        if not manifest:
            return []

        # Detect IIIF version and extract images
        if self._is_iiif_v3(manifest):
            return self._extract_images_v3(manifest)
        else:
            return self._extract_images_v2(manifest)

    def _fetch_manifest(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Fetch and cache a IIIF manifest.

        Args:
            url: Manifest URL

        Returns:
            Parsed manifest JSON or None on error
        """
        # Check cache
        with self._cache_lock:
            if url in self._cache:
                manifest, cached_time = self._cache[url]
                if time.time() - cached_time < self.cache_ttl:
                    logger.debug(f"Cache hit for manifest: {url}")
                    return manifest

        try:
            logger.info(f"Fetching IIIF manifest: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            manifest = response.json()

            # Cache the result
            with self._cache_lock:
                # Evict old entries if cache is full
                if len(self._cache) >= self.max_cache_size:
                    oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                    del self._cache[oldest_key]

                self._cache[url] = (manifest, time.time())

            return manifest

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch manifest {url}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in manifest {url}: {e}")
            return None

    def _is_iiif_v3(self, manifest: Dict[str, Any]) -> bool:
        """Check if manifest is IIIF Presentation API 3.0"""
        context = manifest.get("@context", "")
        if isinstance(context, list):
            return any("presentation/3" in str(c) for c in context)
        return "presentation/3" in str(context)

    def _extract_images_v2(self, manifest: Dict[str, Any]) -> List[IIIFImageInfo]:
        """
        Extract images from IIIF Presentation API 2.x manifest.

        Structure:
        - manifest.sequences[].canvases[].images[].resource.service
        """
        images = []

        try:
            sequences = manifest.get("sequences", [])
            for sequence in sequences:
                canvases = sequence.get("canvases", [])
                for canvas in canvases:
                    canvas_images = canvas.get("images", [])
                    for img in canvas_images:
                        resource = img.get("resource", {})
                        service = resource.get("service", {})

                        # Handle service as list or dict
                        if isinstance(service, list):
                            service = service[0] if service else {}

                        service_id = service.get("@id", "")
                        if not service_id:
                            # Try using resource @id directly
                            service_id = resource.get("@id", "")
                            if service_id and not service_id.endswith("/info.json"):
                                # Extract base URL from image URL
                                # e.g., .../full/full/0/default.jpg -> .../
                                match = re.match(r"(.+?)/(?:full|square|\d+,\d+)/", service_id)
                                if match:
                                    service_id = match.group(1)

                        if service_id:
                            images.append(IIIFImageInfo(
                                image_id=service_id.split("/")[-1],
                                service_url=service_id,
                                width=resource.get("width") or canvas.get("width"),
                                height=resource.get("height") or canvas.get("height"),
                                format=service.get("format", "jpg").split("/")[-1],
                                label=self._get_label_v2(canvas)
                            ))
        except Exception as e:
            logger.warning(f"Error extracting images from v2 manifest: {e}")

        return images

    def _extract_images_v3(self, manifest: Dict[str, Any]) -> List[IIIFImageInfo]:
        """
        Extract images from IIIF Presentation API 3.0 manifest.

        Structure:
        - manifest.items[].items[].items[].body.service[]
        """
        images = []

        try:
            # items = canvases
            for canvas in manifest.get("items", []):
                # canvas.items = annotation pages
                for anno_page in canvas.get("items", []):
                    # annotation page.items = annotations
                    for annotation in anno_page.get("items", []):
                        body = annotation.get("body", {})

                        # Handle body as list
                        if isinstance(body, list):
                            body = body[0] if body else {}

                        service = body.get("service", [])
                        if isinstance(service, dict):
                            service = [service]

                        for svc in service:
                            service_id = svc.get("id", svc.get("@id", ""))
                            if service_id:
                                images.append(IIIFImageInfo(
                                    image_id=service_id.split("/")[-1],
                                    service_url=service_id,
                                    width=body.get("width") or canvas.get("width"),
                                    height=body.get("height") or canvas.get("height"),
                                    format=body.get("format", "image/jpeg").split("/")[-1],
                                    label=self._get_label_v3(canvas)
                                ))

                        # Fallback: use body.id directly if no service
                        if not service and body.get("id"):
                            body_id = body["id"]
                            # Check if it's an image URL
                            if re.search(r"\.(jpg|jpeg|png|gif|tif|tiff)(\?.*)?$", body_id, re.I):
                                images.append(IIIFImageInfo(
                                    image_id=body_id.split("/")[-1],
                                    service_url=body_id.rsplit("/", 4)[0],  # Remove IIIF params
                                    width=body.get("width"),
                                    height=body.get("height"),
                                    format=body_id.split(".")[-1].split("?")[0],
                                    label=self._get_label_v3(canvas)
                                ))
        except Exception as e:
            logger.warning(f"Error extracting images from v3 manifest: {e}")

        return images

    def _get_label_v2(self, obj: Dict[str, Any]) -> Optional[str]:
        """Extract label from IIIF v2 object"""
        label = obj.get("label")
        if isinstance(label, str):
            return label
        if isinstance(label, dict):
            # {"@value": "...", "@language": "en"}
            return label.get("@value")
        if isinstance(label, list) and label:
            first = label[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                return first.get("@value")
        return None

    def _get_label_v3(self, obj: Dict[str, Any]) -> Optional[str]:
        """Extract label from IIIF v3 object"""
        label = obj.get("label", {})
        if isinstance(label, dict):
            # {"en": ["Label"], "fr": ["Étiquette"]}
            for lang in ["en", "none", "und"]:
                if lang in label:
                    values = label[lang]
                    if values:
                        return values[0] if isinstance(values, list) else values
            # Return first available language
            for values in label.values():
                if values:
                    return values[0] if isinstance(values, list) else values
        return None

    def get_image_info(self, image_service_url: str) -> Optional[Dict[str, Any]]:
        """
        Fetch info.json for an image service.

        Args:
            image_service_url: Base URL of IIIF Image API service

        Returns:
            Parsed info.json or None on error
        """
        info_url = f"{image_service_url.rstrip('/')}/info.json"
        return self._fetch_manifest(info_url)  # Same caching logic

    def clear_cache(self) -> None:
        """Clear the manifest cache"""
        with self._cache_lock:
            self._cache.clear()
        logger.info("IIIF manifest cache cleared")
```

---

### Phase 4: Backend Integration

**Goal**: Integrate image resolution into the RAG answer pipeline

#### 4.1 Modify UniversalRagSystem

**File**: `universal_rag_system.py`

Add the following changes:

```python
# Add imports at top of file
from image_resolver import ImageResolver, ResolvedImage
from iiif_client import IIIFClient
from config.image_schema import DatasetImageConfig, load_image_config, ImageUrlType

class UniversalRagSystem:
    def __init__(self, endpoint_url, config, dataset_id=None):
        # ... existing initialization ...

        # Initialize image resolver if config available
        self.image_resolver = None
        self.iiif_client = None
        self._init_image_support(config)

    def _init_image_support(self, config: dict) -> None:
        """Initialize image resolution support if configured."""
        image_config = load_image_config(config)
        if image_config and image_config.enabled:
            self.image_resolver = ImageResolver(
                sparql_endpoint=self.endpoint_url,
                image_config=image_config
            )

            if image_config.iiif.enabled:
                self.iiif_client = IIIFClient(
                    timeout=image_config.iiif.manifest_timeout,
                    cache_ttl=image_config.iiif.manifest_cache_ttl
                )

            logger.info("Image support enabled for this dataset")
        else:
            logger.info("Image support not configured or disabled")

    def get_images_for_entity(self, entity_uri: str) -> List[Dict[str, Any]]:
        """
        Get images for an entity.

        Args:
            entity_uri: URI of the entity

        Returns:
            List of image dictionaries ready for JSON serialization
        """
        if not self.image_resolver:
            return []

        images = self.image_resolver.get_images_for_entity(entity_uri)

        # Resolve IIIF manifests
        resolved_images = []
        for img in images:
            if img.url_type == ImageUrlType.IIIF_MANIFEST and self.iiif_client:
                try:
                    iiif_images = self.iiif_client.get_images_from_manifest(img.original_url)
                    if iiif_images:
                        # Use first image from manifest
                        iiif_img = iiif_images[0]
                        config = self.image_resolver.config.iiif
                        resolved_images.append({
                            "url": iiif_img.get_full_url(),
                            "thumbnail_url": iiif_img.get_thumbnail_url(
                                size=config.thumbnail_size,
                                quality=config.thumbnail_quality
                            ),
                            "type": "iiif",
                            "source": img.source,
                            "width": iiif_img.width,
                            "height": iiif_img.height,
                            "label": iiif_img.label
                        })
                except Exception as e:
                    logger.warning(f"Error resolving IIIF manifest {img.original_url}: {e}")
            else:
                resolved_images.append(img.to_dict())

        return resolved_images

    def answer_question(self, question, include_wikidata=True, include_images=True):
        """
        Answer a question using the universal RAG system.

        Args:
            question: User's question
            include_wikidata: Include Wikidata context
            include_images: Include images for entities (NEW)

        Returns:
            Dict with answer, sources, and images
        """
        # ... existing code up to source preparation ...

        # Prepare sources
        sources = []
        for i, doc in enumerate(retrieved_docs):
            entity_uri = doc.id
            entity_label = doc.metadata.get("label", entity_uri.split('/')[-1])
            raw_triples = doc.metadata.get("raw_triples", [])

            source = {
                "id": i,
                "entity_uri": entity_uri,
                "entity_label": entity_label,
                "type": "graph",
                "entity_type": doc.metadata.get("type", "unknown"),
                "raw_triples": raw_triples
            }

            # Add images if enabled
            if include_images and self.image_resolver:
                source["images"] = self.get_images_for_entity(entity_uri)

            sources.append(source)

        # Add Wikidata sources (with images from P18)
        for entity_info in entities_with_wikidata:
            wikidata_source = {
                "id": f"wikidata_{entity_info['wikidata_id']}",
                "entity_uri": entity_info["entity_uri"],
                "entity_label": entity_info["entity_label"],
                "type": "wikidata",
                "wikidata_id": entity_info["wikidata_id"],
                "wikidata_url": f"https://www.wikidata.org/wiki/{entity_info['wikidata_id']}"
            }

            # Add Wikidata image if available
            if include_images:
                wikidata_data = self.fetch_wikidata_info(entity_info["wikidata_id"])
                if wikidata_data and "properties" in wikidata_data:
                    image_prop = wikidata_data["properties"].get("image")
                    if image_prop:
                        wikidata_source["images"] = [{
                            "url": f"https://commons.wikimedia.org/wiki/Special:FilePath/{image_prop}",
                            "thumbnail_url": f"https://commons.wikimedia.org/wiki/Special:FilePath/{image_prop}?width=200",
                            "type": "wikimedia_commons",
                            "source": "wikidata"
                        }]

            sources.append(wikidata_source)

        return {
            "answer": answer,
            "sources": sources
        }
```

#### 4.2 Add API Endpoint for Images

**File**: `main.py`

Add new endpoint for fetching entity images:

```python
@app.route('/api/entity/<path:entity_uri>/images', methods=['GET'])
def get_entity_images(entity_uri):
    """
    Get images for a specific entity.

    Query Parameters:
        dataset_id: Dataset identifier (required in multi-dataset mode)
        max: Maximum number of images (optional)
    """
    dataset_id = request.args.get('dataset_id')
    max_images = request.args.get('max', type=int)

    # Get RAG system for dataset
    if dataset_id:
        rag_system = dataset_manager.get_dataset(dataset_id)
    else:
        rag_system = default_rag_system

    if not rag_system:
        return jsonify({"error": "Dataset not found or not initialized"}), 404

    try:
        # Decode URI if needed
        from urllib.parse import unquote
        entity_uri = unquote(entity_uri)

        images = rag_system.get_images_for_entity(entity_uri)

        if max_images:
            images = images[:max_images]

        return jsonify({
            "entity_uri": entity_uri,
            "images": images,
            "count": len(images)
        })
    except Exception as e:
        logger.error(f"Error fetching images for {entity_uri}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/datasets/<dataset_id>/images/config', methods=['GET'])
def get_dataset_image_config(dataset_id):
    """Get image configuration for a dataset."""
    config = dataset_manager.get_image_config(dataset_id)

    if not config:
        return jsonify({
            "enabled": False,
            "message": "Images not configured for this dataset"
        })

    return jsonify({
        "enabled": config.enabled,
        "iiif_enabled": config.iiif.enabled,
        "max_images_per_entity": config.display.max_images_per_entity,
        "thumbnail_width": config.display.thumbnail_width,
        "show_in_sources": config.display.show_in_sources,
        "lightbox_enabled": config.display.lightbox_enabled
    })
```

---

### Phase 5: Frontend Display

**Goal**: Display images in the chat interface

#### 5.1 Modify chat.js

**File**: `static/js/chat.js`

```javascript
// Add new function to render images
function renderSourceImages(images, entityLabel) {
    if (!images || images.length === 0) {
        return '';
    }

    let html = '<div class="source-images">';

    images.forEach((img, index) => {
        const thumbnailUrl = img.thumbnail_url || img.url;
        const fullUrl = img.url;
        const altText = img.label || entityLabel || 'Image';
        const sourceType = img.source === 'wikidata' ? 'Wikidata' : 'Graph';

        html += `
            <div class="source-image-container" data-index="${index}">
                <a href="${fullUrl}"
                   target="_blank"
                   class="source-image-link"
                   data-lightbox="entity-${entityLabel}"
                   title="${altText}">
                    <img src="${thumbnailUrl}"
                         alt="${altText}"
                         class="source-thumbnail"
                         loading="lazy"
                         onerror="this.parentElement.style.display='none'">
                </a>
                <span class="image-source-badge">${sourceType}</span>
            </div>
        `;
    });

    html += '</div>';
    return html;
}

// Modify the source rendering in addAssistantMessage function
// In the internalSources.forEach loop, after the existing content:

internalSources.forEach(source => {
    if (source.type === "graph") {
        const tripleCount = source.raw_triples ? source.raw_triples.length : 0;
        const triplesBtnHtml = tripleCount > 0 ?
            `<button class="btn btn-sm btn-outline-primary show-triples-btn ms-2"
                    data-source-id="${source.id}">
                <i class="fas fa-code"></i> Show statements (${tripleCount})
            </button>` : '';

        // NEW: Render images if available
        const imagesHtml = renderSourceImages(source.images, source.entity_label);

        contentHtml += `<li class="source-item graph-source" data-source-id="${source.id}">
            <i class="fas fa-project-diagram me-1"></i>
            <strong>Graph:</strong> ${source.entity_label}
            <br/>
            <small class="text-muted ms-3">URI: <code>${source.entity_uri}</code></small>
            ${triplesBtnHtml}
            ${imagesHtml}
        </li>`;
    }
    // ... rest of source types ...
});

// For Wikidata sources, also add images:
externalSources.forEach(source => {
    const imagesHtml = renderSourceImages(source.images, source.entity_label);

    contentHtml += `<li class="source-item wikidata-source">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Wikidata-logo.svg/20px-Wikidata-logo.svg.png" class="wikidata-logo" alt="Wikidata">
        <a href="${source.wikidata_url}" target="_blank">
            ${source.entity_label} (Wikidata)
        </a>
        <button class="btn btn-sm btn-outline-success show-wikidata-btn"
                data-entity="${source.entity_uri}"
                data-wikidata-id="${source.wikidata_id}">
            <i class="fas fa-info-circle"></i> More info
        </button>
        ${imagesHtml}
    </li>`;
});

// REMOVE the skip for images in Wikidata properties (lines 425-427):
// OLD CODE TO REMOVE:
// if (propName === 'image') {
//     continue;
// }
```

#### 5.2 Add CSS Styles

**File**: `static/css/chat.css`

```css
/* Image Gallery Styles */
.source-images {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
    padding: 8px;
    background-color: rgba(0, 0, 0, 0.02);
    border-radius: 4px;
}

.source-image-container {
    position: relative;
    display: inline-block;
}

.source-image-link {
    display: block;
    text-decoration: none;
    border-radius: 4px;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.source-image-link:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.source-thumbnail {
    max-width: 150px;
    max-height: 100px;
    width: auto;
    height: auto;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid #ddd;
    background-color: #f5f5f5;
}

.source-thumbnail[loading="lazy"] {
    min-width: 100px;
    min-height: 75px;
}

.image-source-badge {
    position: absolute;
    bottom: 4px;
    right: 4px;
    font-size: 0.65rem;
    padding: 2px 6px;
    background-color: rgba(19, 41, 75, 0.85);
    color: white;
    border-radius: 3px;
    text-transform: uppercase;
}

/* Loading placeholder */
.source-thumbnail:not([src]),
.source-thumbnail[src=""] {
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* Lightbox overlay (if not using a library) */
.image-lightbox {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.9);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
    cursor: pointer;
}

.image-lightbox img {
    max-width: 90%;
    max-height: 90%;
    object-fit: contain;
}

.image-lightbox .close-btn {
    position: absolute;
    top: 20px;
    right: 30px;
    color: white;
    font-size: 2rem;
    cursor: pointer;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .source-images {
        justify-content: center;
    }

    .source-thumbnail {
        max-width: 120px;
        max-height: 80px;
    }
}
```

#### 5.3 Optional: Add Lightbox Library

For a better image viewing experience, consider adding a lightweight lightbox library.

**Option A: Simple custom lightbox** (add to chat.js):

```javascript
// Simple lightbox functionality
function setupLightbox() {
    document.addEventListener('click', function(e) {
        const imageLink = e.target.closest('.source-image-link');
        if (imageLink && e.target.classList.contains('source-thumbnail')) {
            e.preventDefault();

            const fullUrl = imageLink.href;
            const alt = e.target.alt;

            // Create lightbox
            const lightbox = document.createElement('div');
            lightbox.className = 'image-lightbox';
            lightbox.innerHTML = `
                <span class="close-btn">&times;</span>
                <img src="${fullUrl}" alt="${alt}">
            `;

            document.body.appendChild(lightbox);
            document.body.style.overflow = 'hidden';

            // Close on click
            lightbox.addEventListener('click', function() {
                lightbox.remove();
                document.body.style.overflow = '';
            });
        }
    });
}

// Call in DOMContentLoaded
document.addEventListener('DOMContentLoaded', function() {
    // ... existing code ...
    setupLightbox();
});
```

**Option B: Use GLightbox library** (recommended for production):

Add to `templates/chat.html`:
```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/glightbox/dist/css/glightbox.min.css">
<script src="https://cdn.jsdelivr.net/npm/glightbox/dist/js/glightbox.min.js"></script>
```

Initialize in chat.js:
```javascript
// After adding assistant message, refresh lightbox
const lightbox = GLightbox({
    selector: '.source-image-link',
    touchNavigation: true,
    loop: true
});
```

---

### Phase 6: Document Generation Enhancement

**Goal**: Store image metadata during document generation for faster retrieval

**File**: `scripts/bulk_generate_documents.py`

This phase is optional but improves performance by pre-computing image URLs during document generation.

```python
# Add to BulkDocumentGenerator class

def extract_entity_images(self, entity_uri: str) -> List[Dict[str, Any]]:
    """
    Extract image information for an entity.

    Args:
        entity_uri: URI of the entity

    Returns:
        List of image info dictionaries
    """
    if not self.image_config or not self.image_config.enabled:
        return []

    images = []

    for predicate_config in self.image_config.get_sorted_predicates():
        query = f"""
        {self.prefixes}

        SELECT DISTINCT ?value WHERE {{
            <{entity_uri}> {predicate_config.path} ?value .
            {f"FILTER({predicate_config.filter})" if predicate_config.filter else ""}
        }}
        LIMIT 5
        """

        try:
            results = self.execute_query(query)
            for binding in results:
                value = binding.get("value", {}).get("value")
                if value:
                    images.append({
                        "url": value,
                        "type": predicate_config.type.value,
                        "predicate": predicate_config.path
                    })
        except Exception as e:
            logger.warning(f"Error extracting images for {entity_uri}: {e}")

    return images

def create_document(self, entity_uri: str, context_depth: int = 2) -> tuple:
    """Create document for an entity. Returns (text, label, types, metadata)."""
    # ... existing code ...

    # Extract images
    images = self.extract_entity_images(entity_uri)

    # Return with additional metadata
    return text, entity_label, entity_types, {
        "images": images,
        "wikidata_id": self.get_wikidata_id(entity_uri)
    }
```

The extracted images are then stored in the GraphDocument metadata during graph building, making runtime image resolution optional (only needed for IIIF manifest parsing).

---

## API Specifications

### GET /api/chat (Modified)

**Request**:
```json
{
    "question": "What artworks depict the Virgin Mary?",
    "dataset_id": "mah",
    "include_images": true
}
```

**Response**:
```json
{
    "answer": "Several artworks in the collection depict the Virgin Mary...",
    "sources": [
        {
            "id": 0,
            "entity_uri": "https://data.mahmah.ch/work/1415339",
            "entity_label": "La Vierge à l'Enfant...",
            "type": "graph",
            "entity_type": "Human-Made Object",
            "raw_triples": [...],
            "images": [
                {
                    "url": "https://iiif.hedera.unige.ch/iiif/2/mah/1415339/full/max/0/default.jpg",
                    "thumbnail_url": "https://iiif.hedera.unige.ch/iiif/2/mah/1415339/full/200,/0/default.jpg",
                    "type": "iiif",
                    "source": "graph",
                    "width": 2000,
                    "height": 1500,
                    "label": "La Vierge à l'Enfant"
                }
            ]
        },
        {
            "id": "wikidata_Q12345",
            "entity_uri": "...",
            "entity_label": "...",
            "type": "wikidata",
            "wikidata_id": "Q12345",
            "wikidata_url": "https://www.wikidata.org/wiki/Q12345",
            "images": [
                {
                    "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Example.jpg",
                    "thumbnail_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Example.jpg?width=200",
                    "type": "wikimedia_commons",
                    "source": "wikidata"
                }
            ]
        }
    ]
}
```

### GET /api/entity/{uri}/images

**Request**:
```
GET /api/entity/https%3A%2F%2Fdata.mahmah.ch%2Fwork%2F1415339/images?dataset_id=mah&max=5
```

**Response**:
```json
{
    "entity_uri": "https://data.mahmah.ch/work/1415339",
    "images": [
        {
            "url": "https://iiif.hedera.unige.ch/iiif/2/mah/1415339/full/max/0/default.jpg",
            "thumbnail_url": "https://iiif.hedera.unige.ch/iiif/2/mah/1415339/full/200,/0/default.jpg",
            "type": "iiif",
            "source": "graph",
            "width": 2000,
            "height": 1500
        }
    ],
    "count": 1
}
```

### GET /api/datasets/{id}/images/config

**Response**:
```json
{
    "enabled": true,
    "iiif_enabled": true,
    "max_images_per_entity": 3,
    "thumbnail_width": 150,
    "show_in_sources": true,
    "lightbox_enabled": true
}
```

---

## Security Considerations

### 1. URL Validation

```python
# In image_resolver.py

import re
from urllib.parse import urlparse

ALLOWED_SCHEMES = {'http', 'https'}
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.tiff', '.tif', '.webp'}

def is_safe_image_url(url: str) -> bool:
    """Validate that a URL is safe to use as an image source."""
    try:
        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme.lower() not in ALLOWED_SCHEMES:
            return False

        # Check for local/private IPs
        hostname = parsed.hostname
        if hostname:
            # Block localhost, private IPs
            if hostname in ('localhost', '127.0.0.1', '0.0.0.0'):
                return False
            if hostname.startswith('192.168.') or hostname.startswith('10.'):
                return False
            if hostname.startswith('172.') and 16 <= int(hostname.split('.')[1]) <= 31:
                return False

        # Check path doesn't contain suspicious patterns
        if '..' in parsed.path or '\\' in parsed.path:
            return False

        return True
    except Exception:
        return False
```

### 2. Content-Type Verification

When proxying images (if implemented), verify Content-Type:

```python
ALLOWED_CONTENT_TYPES = {
    'image/jpeg', 'image/png', 'image/gif',
    'image/tiff', 'image/webp', 'application/json'  # For IIIF
}

def verify_image_content_type(response: requests.Response) -> bool:
    content_type = response.headers.get('Content-Type', '').split(';')[0].strip()
    return content_type in ALLOWED_CONTENT_TYPES
```

### 3. CORS Considerations

Many image servers have CORS restrictions. Options:
1. Use image proxy endpoint (adds server load)
2. Display images that allow CORS
3. Use `crossorigin="anonymous"` attribute (for CORS-enabled servers)

```html
<img src="${thumbnailUrl}"
     crossorigin="anonymous"
     onerror="this.removeAttribute('crossorigin'); this.src=this.src;">
```

### 4. Rate Limiting

Implement rate limiting for image endpoints:

```python
from functools import wraps
from flask import request, jsonify
import time

# Simple in-memory rate limiter
rate_limit_cache = {}

def rate_limit(max_requests: int, window_seconds: int):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            key = f"{request.remote_addr}:{request.endpoint}"
            now = time.time()

            # Clean old entries
            rate_limit_cache[key] = [
                t for t in rate_limit_cache.get(key, [])
                if now - t < window_seconds
            ]

            if len(rate_limit_cache.get(key, [])) >= max_requests:
                return jsonify({"error": "Rate limit exceeded"}), 429

            rate_limit_cache.setdefault(key, []).append(now)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Usage
@app.route('/api/entity/<path:entity_uri>/images')
@rate_limit(max_requests=30, window_seconds=60)
def get_entity_images(entity_uri):
    # ...
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/test_image_resolver.py`

```python
import pytest
from image_resolver import ImageResolver, ResolvedImage
from config.image_schema import DatasetImageConfig, PredicateConfig, ImageUrlType

class TestImageResolver:

    @pytest.fixture
    def mock_config(self):
        return DatasetImageConfig(
            enabled=True,
            predicates=[
                PredicateConfig(
                    path="crm:P138_has_representation",
                    type=ImageUrlType.DIRECT,
                    filter="CONTAINS(STR(?value), '.jpg')"
                )
            ]
        )

    def test_wikimedia_thumbnail_generation(self):
        resolver = ImageResolver("http://test/sparql", self.mock_config)
        url = "http://commons.wikimedia.org/wiki/Special:FilePath/Test.jpg"

        img = ResolvedImage(original_url=url, url_type=ImageUrlType.WIKIMEDIA_COMMONS)
        resolver._resolve_thumbnail(img)

        assert "?width=200" in img.thumbnail_url

    def test_iiif_thumbnail_generation(self):
        resolver = ImageResolver("http://test/sparql", self.mock_config)
        url = "https://iiif.example.org/image/123/full/max/0/default.jpg"

        img = ResolvedImage(original_url=url, url_type=ImageUrlType.IIIF_IMAGE)
        resolver._resolve_thumbnail(img)

        assert "/200,/" in img.thumbnail_url


class TestIIIFClient:

    def test_iiif_v2_manifest_parsing(self, iiif_client, sample_manifest_v2):
        images = iiif_client._extract_images_v2(sample_manifest_v2)
        assert len(images) > 0
        assert images[0].service_url is not None

    def test_iiif_v3_manifest_parsing(self, iiif_client, sample_manifest_v3):
        images = iiif_client._extract_images_v3(sample_manifest_v3)
        assert len(images) > 0
```

### Integration Tests

```python
class TestImageIntegration:

    def test_answer_includes_images(self, client, mah_dataset):
        response = client.post('/api/chat', json={
            "question": "Show me artworks",
            "dataset_id": "mah",
            "include_images": True
        })

        data = response.get_json()
        assert response.status_code == 200

        # Check at least one source has images
        sources_with_images = [s for s in data['sources'] if s.get('images')]
        assert len(sources_with_images) > 0

    def test_image_endpoint(self, client, mah_dataset):
        entity_uri = "https://data.mahmah.ch/work/1415339"
        response = client.get(
            f'/api/entity/{quote(entity_uri)}/images',
            query_string={'dataset_id': 'mah'}
        )

        assert response.status_code == 200
        data = response.get_json()
        assert 'images' in data
```

### Manual Testing Checklist

- [ ] Images display correctly in sources section
- [ ] Thumbnails load with acceptable latency (<2s)
- [ ] Full-size images open in lightbox
- [ ] Broken images fail gracefully (no error shown to user)
- [ ] Wikidata images display correctly
- [ ] IIIF images display correctly
- [ ] Direct image URLs display correctly
- [ ] Images disabled dataset shows no images
- [ ] Mobile responsive layout works
- [ ] No CORS errors in console

---

## Migration Guide

### Migrating Existing Deployments

1. **Update configuration files**:
   ```bash
   # Add image config to datasets.yaml
   cp config/datasets.yaml config/datasets.yaml.backup
   # Edit datasets.yaml to add images section for each dataset
   ```

2. **Install new dependencies** (if any):
   ```bash
   pip install pydantic  # If not already installed
   ```

3. **Create new module files**:
   ```bash
   touch config/image_schema.py
   touch config/namespaces.py
   touch image_resolver.py
   touch iiif_client.py
   ```

4. **Update frontend files**:
   ```bash
   # Backup existing files
   cp static/js/chat.js static/js/chat.js.backup
   cp static/css/chat.css static/css/chat.css.backup

   # Apply changes (manually or via patch)
   ```

5. **Optional: Rebuild document cache** (for Phase 6):
   ```bash
   # Regenerate documents with image metadata
   python main.py --env .env.local --dataset mah --rebuild --process-only
   ```

6. **Test**:
   ```bash
   # Start server
   python main.py --env .env.local

   # Test image endpoint
   curl "http://localhost:5001/api/entity/https%3A%2F%2Fdata.mahmah.ch%2Fwork%2F1415339/images?dataset_id=mah"
   ```

### Rollback Procedure

If issues occur:

1. Restore configuration:
   ```bash
   cp config/datasets.yaml.backup config/datasets.yaml
   ```

2. Restore frontend:
   ```bash
   cp static/js/chat.js.backup static/js/chat.js
   cp static/css/chat.css.backup static/css/chat.css
   ```

3. Restart server

---

## Appendix: Sample IIIF Manifests

### IIIF Presentation API 2.1

```json
{
  "@context": "http://iiif.io/api/presentation/2/context.json",
  "@id": "https://iiif.hedera.unige.ch/iiif/manifests/mah/30085940",
  "@type": "sc:Manifest",
  "label": "Artwork Title",
  "sequences": [
    {
      "@type": "sc:Sequence",
      "canvases": [
        {
          "@id": "https://iiif.hedera.unige.ch/canvas/1",
          "@type": "sc:Canvas",
          "label": "Page 1",
          "width": 2000,
          "height": 1500,
          "images": [
            {
              "@type": "oa:Annotation",
              "resource": {
                "@id": "https://iiif.hedera.unige.ch/iiif/2/mah/30085940/full/full/0/default.jpg",
                "@type": "dctypes:Image",
                "width": 2000,
                "height": 1500,
                "service": {
                  "@context": "http://iiif.io/api/image/2/context.json",
                  "@id": "https://iiif.hedera.unige.ch/iiif/2/mah/30085940",
                  "profile": "http://iiif.io/api/image/2/level2.json"
                }
              }
            }
          ]
        }
      ]
    }
  ]
}
```

### IIIF Presentation API 3.0

```json
{
  "@context": "http://iiif.io/api/presentation/3/context.json",
  "id": "https://iiif.hedera.unige.ch/iiif/manifests/mah/30085940",
  "type": "Manifest",
  "label": { "en": ["Artwork Title"] },
  "items": [
    {
      "id": "https://iiif.hedera.unige.ch/canvas/1",
      "type": "Canvas",
      "label": { "en": ["Page 1"] },
      "width": 2000,
      "height": 1500,
      "items": [
        {
          "id": "https://iiif.hedera.unige.ch/page/1",
          "type": "AnnotationPage",
          "items": [
            {
              "id": "https://iiif.hedera.unige.ch/annotation/1",
              "type": "Annotation",
              "motivation": "painting",
              "body": {
                "id": "https://iiif.hedera.unige.ch/iiif/2/mah/30085940/full/max/0/default.jpg",
                "type": "Image",
                "format": "image/jpeg",
                "width": 2000,
                "height": 1500,
                "service": [
                  {
                    "id": "https://iiif.hedera.unige.ch/iiif/2/mah/30085940",
                    "type": "ImageService2",
                    "profile": "level2"
                  }
                ]
              },
              "target": "https://iiif.hedera.unige.ch/canvas/1"
            }
          ]
        }
      ]
    }
  ]
}
```

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-26 | 1.0 | Initial plan created |

---

## References

- [IIIF Presentation API 3.0](https://iiif.io/api/presentation/3.0/)
- [IIIF Image API 3.0](https://iiif.io/api/image/3.0/)
- [CIDOC-CRM Documentation](https://www.cidoc-crm.org/versions-of-the-cidoc-crm)
- [Wikimedia Commons API](https://commons.wikimedia.org/wiki/Commons:API)
