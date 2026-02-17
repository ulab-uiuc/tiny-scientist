const SECTION_DEFINITIONS = [
  { title: 'Problem', regex: /\*\*Problem:\*\*|Problem:|Problem\s*\*\*|\*\*Description:\*\*|Description:/ },
  { title: 'Approach', regex: /\*\*Approach:\*\*|Approach:|Approach\s*\*\*/ },
  { title: 'Experiment', regex: /\*\*Experiment:\*\*|Experiment:|Experiment\s*\*\*/ },
  { title: 'Impact', regex: /\*\*Impact:\*\*|Impact:|Impact\s*\*\*/ },
  { title: 'Feasibility', regex: /\*\*Feasibility:\*\*|Feasibility:|Feasibility\s*\*\*/ },
  { title: 'Novelty', regex: /\*\*Novelty Comparison:\*\*|Novelty Comparison:|Novelty:|Novelty\s*\*\*/ }
];


export function extractContentSections(content = '') {
  if (!content) {
    return [];
  }

  const parsedSections = [];
  const allMatches = [];

  SECTION_DEFINITIONS.forEach((section) => {
    const displayTitle = section.title;
    let match;
    let tempContent = content;
    let offset = 0;

    while ((match = tempContent.match(section.regex)) !== null) {
      const startPos = match.index + offset;
      const matchLength = match[0].length;

      allMatches.push({
        title: displayTitle,
        position: startPos,
        length: matchLength
      });

      offset += match.index + matchLength;
      tempContent = content.substring(offset);
    }
  });

  allMatches.sort((a, b) => a.position - b.position);

  for (let i = 0; i < allMatches.length; i++) {
    const currentMatch = allMatches[i];
    const startPos = currentMatch.position + currentMatch.length;
    let endPos = content.length;

    if (i < allMatches.length - 1) {
      endPos = allMatches[i + 1].position;
    }

    const sectionContent = content.substring(startPos, endPos).trim();

    if (sectionContent) {
      parsedSections.push({
        title: currentMatch.title,
        content: sectionContent
      });
    }
  }

  return parsedSections;
}

export function buildNodeContent(node = {}) {
  if (!node || typeof node !== 'object') return {};

  const source =
    node.originalData && typeof node.originalData === 'object'
      ? node.originalData
      : null;
  const idea = source ? { ...source } : {};

  const pick = (...values) => {
    for (const value of values) {
      if (value !== undefined && value !== null && value !== '') return value;
    }
    return undefined;
  };

  const id = pick(idea.id, idea.ID, node.id);
  const title = pick(idea.title, idea.Title, idea.Name, node.title);
  const content = pick(idea.content, idea.Problem, node.content);

  if (id !== undefined) idea.id = id;
  if (title !== undefined) idea.title = title;
  if (content !== undefined) idea.content = content;

  if (node.scores && idea.scores == null) {
    idea.scores = node.scores;
  }
  const problemHighlights = pick(
    idea.problemHighlights,
    idea.problem_highlights,
    node.problemHighlights
  );
  if (problemHighlights !== undefined) {
    idea.problemHighlights = problemHighlights;
  }

  const dimension1Score = pick(
    idea.Dimension1Score,
    idea.dimension1Score,
    node.dimension1Score,
    idea.impactScore,
    node.impactScore
  );
  const dimension2Score = pick(
    idea.Dimension2Score,
    idea.dimension2Score,
    node.dimension2Score,
    idea.feasibilityScore,
    node.feasibilityScore
  );
  const dimension3Score = pick(
    idea.Dimension3Score,
    idea.dimension3Score,
    node.dimension3Score,
    idea.noveltyScore,
    node.noveltyScore
  );

  const dimension1Reason = pick(
    idea.Dimension1Reason,
    idea.dimension1Reason,
    idea.ImpactReason,
    node.evaluationReasoning?.impactReasoning
  );
  const dimension2Reason = pick(
    idea.Dimension2Reason,
    idea.dimension2Reason,
    idea.FeasibilityReason,
    node.evaluationReasoning?.feasibilityReasoning
  );
  const dimension3Reason = pick(
    idea.Dimension3Reason,
    idea.dimension3Reason,
    idea.NoveltyReason,
    node.evaluationReasoning?.noveltyReasoning
  );

  if (dimension1Score !== undefined) idea['dimension1'] = dimension1Score;
  if (dimension2Score !== undefined) idea['dimension2'] = dimension2Score;
  if (dimension3Score !== undefined) idea['dimension3'] = dimension3Score;
  if (dimension1Reason !== undefined) idea['dimension1-reason'] = dimension1Reason;
  if (dimension2Reason !== undefined) idea['dimension2-reason'] = dimension2Reason;
  if (dimension3Reason !== undefined) idea['dimension3-reason'] = dimension3Reason;

  [
    'ID',
    'Title',
    'Name',
    'Problem',
    'Content',
    'ImpactReason',
    'FeasibilityReason',
    'NoveltyReason',
    'Dimension1Reason',
    'Dimension2Reason',
    'Dimension3Reason',
    'Dimension1Score',
    'Dimension2Score',
    'Dimension3Score',
    'dimension1Score',
    'dimension2Score',
    'dimension3Score',
    'impactScore',
    'feasibilityScore',
    'noveltyScore',
    'evaluationReasoning',
    'problem_highlights'
  ].forEach((key) => {
    if (key in idea) delete idea[key];
  });

  return idea;
}
