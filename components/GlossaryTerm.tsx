
import React from 'react';

interface GlossaryTermProps {
    term: string;
    definition: string;
}

export const GlossaryTerm: React.FC<GlossaryTermProps> = ({ term, definition }) => (
    <span className="glossary-term">{term}<span className="tooltip">{definition}</span></span>
);
