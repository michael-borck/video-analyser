"""Default example rubrics for common presentation types."""

from video_lens.analysis.rubric_system import Rubric, RubricBuilder


def create_academic_presentation_rubric() -> Rubric:
    """Create a rubric for academic presentations (conferences, seminars)."""
    builder = RubricBuilder(
        name="Academic Presentation",
        description="Evaluate academic presentations at conferences, seminars, or classroom settings",
    )

    # Content Category
    content = builder.add_category(
        name="Content Quality",
        description="Evaluation of presentation content",
        weight=2.0,
    )
    content.add_criterion(
        name="Research Depth",
        description="Demonstrates thorough understanding of research methodology and findings",
        weight=1.5,
        scoring_guide="1=Superficial, 3=Adequate, 5=Comprehensive and insightful",
    )
    content.add_criterion(
        name="Clear Thesis/Purpose",
        description="Main research question or contribution is clearly stated",
        weight=1.5,
        scoring_guide="1=Unclear, 3=Moderately clear, 5=Crystal clear",
    )
    content.add_criterion(
        name="Supporting Evidence",
        description="Uses relevant data, citations, and examples to support claims",
        weight=1.0,
        scoring_guide="1=Minimal, 3=Adequate citations, 5=Excellent sourcing",
    )

    # Delivery Category
    delivery = builder.add_category(
        name="Delivery & Presentation",
        description="Evaluation of speaking and visual presentation",
        weight=1.5,
    )
    delivery.add_criterion(
        name="Speaking Clarity",
        description="Speech is clear, at appropriate pace, and easy to understand",
        weight=1.0,
        scoring_guide="1=Hard to follow, 3=Generally clear, 5=Excellent clarity",
    )
    delivery.add_criterion(
        name="Engagement",
        description="Maintains audience attention through tone, eye contact, and enthusiasm",
        weight=1.0,
        scoring_guide="1=Monotone, 3=Adequate, 5=Highly engaging",
    )
    delivery.add_criterion(
        name="Visual Aids",
        description="Slides/visuals are clear, professional, and enhance understanding",
        weight=0.5,
        scoring_guide="1=Distracting, 3=Adequate, 5=Professional and effective",
    )

    # Organization Category
    org = builder.add_category(
        name="Organization & Structure",
        description="Logical flow and coherent presentation structure",
        weight=1.0,
    )
    org.add_criterion(
        name="Logical Flow",
        description="Ideas presented in logical sequence that's easy to follow",
        weight=1.0,
        scoring_guide="1=Disorganized, 3=Generally logical, 5=Excellent flow",
    )
    org.add_criterion(
        name="Time Management",
        description="Appropriately uses allocated time; covers material without rushing or excess",
        weight=0.5,
        scoring_guide="1=Way over/under time, 3=Slightly off, 5=Perfect timing",
    )

    # Set custom scoring scale
    builder.set_scoring_scale(
        min_score=1,
        max_score=100,
        labels={
            1: "Poor - Needs significant improvement",
            20: "Below Average - Notable deficiencies",
            40: "Average - Meets basic expectations",
            60: "Good - Exceeds expectations",
            80: "Very Good - Strong performance",
            100: "Excellent - Outstanding work",
        },
    )

    return builder.build(is_template=True, tags=["academic", "conference", "research"])


def create_business_pitch_rubric() -> Rubric:
    """Create a rubric for business pitches and startup presentations."""
    builder = RubricBuilder(
        name="Business Pitch",
        description="Evaluate business pitches, startup presentations, and investor pitches",
    )

    # Business Model Category
    business = builder.add_category(
        name="Business Model & Value Prop",
        description="Clarity of business model and value proposition",
        weight=2.0,
    )
    business.add_criterion(
        name="Clear Problem Statement",
        description="Identifies and articulates the problem being solved",
        weight=1.0,
        scoring_guide="1=Vague problem, 3=Clear problem, 5=Compelling problem",
    )
    business.add_criterion(
        name="Unique Value Proposition",
        description="Clearly explains what makes the solution unique and valuable",
        weight=1.5,
        scoring_guide="1=Generic, 3=Adequate differentiation, 5=Clear competitive advantage",
    )
    business.add_criterion(
        name="Market Understanding",
        description="Demonstrates knowledge of target market, size, and opportunity",
        weight=1.0,
        scoring_guide="1=No market analysis, 3=Basic market data, 5=In-depth market insights",
    )

    # Financial Category
    financial = builder.add_category(
        name="Financial Viability",
        description="Business financials and sustainability",
        weight=1.5,
    )
    financial.add_criterion(
        name="Revenue Model",
        description="Clear explanation of how the business makes money",
        weight=1.0,
        scoring_guide="1=Unclear, 3=Reasonable model, 5=Compelling monetization",
    )
    financial.add_criterion(
        name="Financial Projections",
        description="Realistic financial projections and path to profitability",
        weight=1.0,
        scoring_guide="1=No projections, 3=Basic numbers, 5=Detailed, realistic forecasts",
    )

    # Presentation Category
    presentation = builder.add_category(
        name="Presentation Quality",
        description="Delivery, visuals, and persuasiveness",
        weight=1.5,
    )
    presentation.add_criterion(
        name="Persuasiveness",
        description="Effectively persuades audience of business potential",
        weight=1.0,
        scoring_guide="1=Unconvincing, 3=Moderately persuasive, 5=Highly compelling",
    )
    presentation.add_criterion(
        name="Confidence & Credibility",
        description="Presenter demonstrates confidence and credibility in the idea",
        weight=0.75,
        scoring_guide="1=Uncertain, 3=Confident, 5=Highly credible and poised",
    )
    presentation.add_criterion(
        name="Visual Design",
        description="Pitch deck is professional, clean, and visually appealing",
        weight=0.75,
        scoring_guide="1=Poor design, 3=Adequate, 5=Professional, polished design",
    )

    builder.set_scoring_scale(
        min_score=1,
        max_score=100,
        labels={
            1: "Poor - Not investment-ready",
            20: "Below Average - Significant concerns",
            40: "Average - Interesting but needs work",
            60: "Good - Strong pitch",
            80: "Very Good - Compelling investment opportunity",
            100: "Excellent - Ready to pitch to investors",
        },
    )

    return builder.build(
        is_template=True, tags=["business", "startup", "pitch", "investor"]
    )


def create_teaching_demo_rubric() -> Rubric:
    """Create a rubric for teaching demonstrations and lesson presentations."""
    builder = RubricBuilder(
        name="Teaching Demonstration",
        description="Evaluate teaching effectiveness, pedagogy, and classroom engagement",
    )

    # Pedagogy Category
    pedagogy = builder.add_category(
        name="Pedagogical Approach",
        description="Teaching methodology and instructional design",
        weight=2.0,
    )
    pedagogy.add_criterion(
        name="Learning Objectives",
        description="Clear learning objectives communicated to students",
        weight=1.0,
        scoring_guide="1=No objectives stated, 3=Objectives mentioned, 5=Clear, measurable objectives",
    )
    pedagogy.add_criterion(
        name="Instructional Design",
        description="Well-structured lesson with clear organization and flow",
        weight=1.5,
        scoring_guide="1=Disorganized, 3=Adequate structure, 5=Excellent instructional design",
    )
    pedagogy.add_criterion(
        name="Active Learning",
        description="Incorporates student engagement and active learning strategies",
        weight=1.0,
        scoring_guide="1=Lecture-only, 3=Some interaction, 5=Highly interactive",
    )

    # Content & Communication Category
    content = builder.add_category(
        name="Content & Communication",
        description="Accuracy, clarity, and depth of content delivery",
        weight=1.5,
    )
    content.add_criterion(
        name="Content Accuracy",
        description="Subject matter is accurate and free of errors",
        weight=0.75,
        scoring_guide="1=Significant errors, 3=Mostly accurate, 5=Fully accurate",
    )
    content.add_criterion(
        name="Explanation Clarity",
        description="Concepts explained clearly and at appropriate level",
        weight=1.0,
        scoring_guide="1=Confusing, 3=Generally clear, 5=Very clear and well-explained",
    )
    content.add_criterion(
        name="Use of Examples",
        description="Effective use of examples to illustrate concepts",
        weight=0.75,
        scoring_guide="1=No examples, 3=Adequate examples, 5=Excellent, relevant examples",
    )

    # Student Engagement Category
    engagement = builder.add_category(
        name="Student Engagement & Classroom Management",
        description="Ability to engage and manage students",
        weight=1.5,
    )
    engagement.add_criterion(
        name="Student Engagement",
        description="Maintains student attention and enthusiasm",
        weight=1.0,
        scoring_guide="1=Students disengaged, 3=Moderate engagement, 5=Highly engaged",
    )
    engagement.add_criterion(
        name="Classroom Dynamics",
        description="Manages classroom effectively; encourages participation",
        weight=1.0,
        scoring_guide="1=Poor management, 3=Adequate management, 5=Excellent dynamics",
    )

    builder.set_scoring_scale(
        min_score=1,
        max_score=100,
        labels={
            1: "Poor - Not effective teaching",
            20: "Below Average - Needs improvement",
            40: "Average - Meets basic standards",
            60: "Good - Effective teaching",
            80: "Very Good - Strong educational impact",
            100: "Excellent - Outstanding educator",
        },
    )

    return builder.build(is_template=True, tags=["teaching", "education", "classroom"])


def create_technical_presentation_rubric() -> Rubric:
    """Create a rubric for technical presentations and software demonstrations."""
    builder = RubricBuilder(
        name="Technical Presentation",
        description="Evaluate technical presentations, software demos, and technical walkthroughs",
    )

    # Technical Content Category
    technical = builder.add_category(
        name="Technical Content",
        description="Accuracy and quality of technical information presented",
        weight=2.5,
    )
    technical.add_criterion(
        name="Technical Accuracy",
        description="Technical information is correct and demonstrates deep understanding",
        weight=1.5,
        scoring_guide="1=Significant errors, 3=Mostly accurate, 5=Completely accurate and authoritative",
    )
    technical.add_criterion(
        name="Technical Depth",
        description="Appropriate depth for audience - not too basic or overly complex",
        weight=1.0,
        scoring_guide="1=Too simplistic or too complex, 3=Adequate depth, 5=Perfect balance for audience",
    )
    technical.add_criterion(
        name="Problem-Solving Approach",
        description="Clear explanation of technical challenges and solutions",
        weight=1.0,
        scoring_guide="1=Unclear approach, 3=Reasonable solutions, 5=Innovative and well-justified solutions",
    )

    # Code/Demo Quality Category
    code_demo = builder.add_category(
        name="Code & Demonstration Quality",
        description="Quality of code examples and live demonstrations",
        weight=2.0,
    )
    code_demo.add_criterion(
        name="Code Clarity",
        description="Code is well-structured, readable, and properly explained",
        weight=1.0,
        scoring_guide="1=Confusing code, 3=Readable code, 5=Excellent code quality and documentation",
    )
    code_demo.add_criterion(
        name="Demonstration Effectiveness",
        description="Live demos work smoothly and effectively illustrate key points",
        weight=1.0,
        scoring_guide="1=Demos fail or don't work, 3=Demos work but could be smoother, 5=Flawless, effective demonstrations",
    )
    code_demo.add_criterion(
        name="Visual Technical Aids",
        description="Diagrams, architecture drawings, and technical visuals are clear and helpful",
        weight=0.5,
        scoring_guide="1=Poor or missing visuals, 3=Adequate visuals, 5=Excellent technical visualizations",
    )

    # Communication Category
    communication = builder.add_category(
        name="Technical Communication",
        description="Ability to explain complex technical concepts clearly",
        weight=1.5,
    )
    communication.add_criterion(
        name="Concept Explanation",
        description="Complex technical concepts explained clearly and accessibly",
        weight=1.0,
        scoring_guide="1=Confusing explanations, 3=Generally clear, 5=Exceptionally clear and insightful",
    )
    communication.add_criterion(
        name="Audience Adaptation",
        description="Presentation adapted appropriately for audience technical level",
        weight=0.75,
        scoring_guide="1=Doesn't consider audience, 3=Some adaptation, 5=Perfectly tailored to audience",
    )

    builder.set_scoring_scale(
        min_score=1,
        max_score=100,
        labels={
            1: "Poor - Technically inadequate",
            20: "Below Average - Notable technical issues",
            40: "Average - Technically competent",
            60: "Good - Strong technical presentation",
            80: "Very Good - Excellent technical skills",
            100: "Excellent - Outstanding technical communication",
        },
    )

    return builder.build(
        is_template=True, tags=["technical", "software", "demo", "code"]
    )


def create_sales_marketing_rubric() -> Rubric:
    """Create a rubric for sales and marketing presentations."""
    builder = RubricBuilder(
        name="Sales & Marketing Presentation",
        description="Evaluate sales pitches, marketing campaigns, and product launch presentations",
    )

    # Product/Market Category
    product_market = builder.add_category(
        name="Product & Market Understanding",
        description="Knowledge of product and target market",
        weight=2.0,
    )
    product_market.add_criterion(
        name="Product Knowledge",
        description="Deep understanding of product features, benefits, and competitive advantages",
        weight=1.0,
        scoring_guide="1=Limited product knowledge, 3=Good product understanding, 5=Expert product mastery",
    )
    product_market.add_criterion(
        name="Market Awareness",
        description="Clear understanding of target market, customer needs, and market dynamics",
        weight=1.0,
        scoring_guide="1=Poor market understanding, 3=Basic market awareness, 5=Excellent market insights",
    )

    # Value Proposition Category
    value_prop = builder.add_category(
        name="Value Proposition & Messaging",
        description="Clarity and persuasiveness of value proposition",
        weight=2.0,
    )
    value_prop.add_criterion(
        name="Unique Selling Proposition",
        description="Clear, compelling articulation of what makes the offering unique",
        weight=1.0,
        scoring_guide="1=Generic messaging, 3=Clear value prop, 5=Highly differentiated USP",
    )
    value_prop.add_criterion(
        name="Customer Benefits",
        description="Effectively communicates specific benefits to target customers",
        weight=1.0,
        scoring_guide="1=Features-focused, 3=Benefits mentioned, 5=Customer-centric benefits",
    )

    # Presentation & Persuasion Category
    presentation = builder.add_category(
        name="Presentation & Persuasion",
        description="Delivery effectiveness and persuasive techniques",
        weight=1.5,
    )
    presentation.add_criterion(
        name="Persuasive Techniques",
        description="Uses effective sales techniques, storytelling, and objection handling",
        weight=1.0,
        scoring_guide="1=Monotone delivery, 3=Some persuasion, 5=Masterful sales presentation",
    )
    presentation.add_criterion(
        name="Call to Action",
        description="Clear, compelling next steps for the audience",
        weight=0.5,
        scoring_guide="1=No clear CTA, 3=Mentioned CTA, 5=Urgent, actionable CTA",
    )

    # Visual & Materials Category
    visuals = builder.add_category(
        name="Visuals & Marketing Materials",
        description="Quality of supporting materials and visuals",
        weight=0.5,
    )
    visuals.add_criterion(
        name="Marketing Materials",
        description="Professional quality of slides, demos, and supporting materials",
        weight=0.5,
        scoring_guide="1=Poor materials, 3=Adequate materials, 5=Professional, polished materials",
    )

    builder.set_scoring_scale(
        min_score=1,
        max_score=100,
        labels={
            1: "Poor - Not sales-ready",
            20: "Below Average - Needs significant work",
            40: "Average - Basic sales competence",
            60: "Good - Strong sales presentation",
            80: "Very Good - Compelling sales skills",
            100: "Excellent - Outstanding sales performance",
        },
    )

    return builder.build(
        is_template=True, tags=["sales", "marketing", "pitch", "product", "campaign"]
    )


def create_legal_presentation_rubric() -> Rubric:
    """Create a rubric for legal presentations and arguments."""
    builder = RubricBuilder(
        name="Legal Presentation",
        description="Evaluate legal arguments, court presentations, and legal advocacy",
    )

    # Legal Analysis Category
    legal_analysis = builder.add_category(
        name="Legal Analysis & Reasoning",
        description="Quality of legal analysis and logical reasoning",
        weight=2.5,
    )
    legal_analysis.add_criterion(
        name="Legal Knowledge",
        description="Demonstrates comprehensive knowledge of relevant laws and precedents",
        weight=1.5,
        scoring_guide="1=Limited legal knowledge, 3=Adequate legal understanding, 5=Expert legal mastery",
    )
    legal_analysis.add_criterion(
        name="Logical Reasoning",
        description="Clear, logical progression of arguments with sound legal reasoning",
        weight=1.0,
        scoring_guide="1=Illogical arguments, 3=Generally logical, 5=Impeccable legal reasoning",
    )

    # Evidence & Facts Category
    evidence = builder.add_category(
        name="Evidence & Factual Support",
        description="Use of evidence and factual support",
        weight=2.0,
    )
    evidence.add_criterion(
        name="Evidence Quality",
        description="Uses relevant, credible evidence to support legal arguments",
        weight=1.0,
        scoring_guide="1=Weak or irrelevant evidence, 3=Adequate evidence, 5=Strong, compelling evidence",
    )
    evidence.add_criterion(
        name="Case Law & Precedents",
        description="Appropriate use of relevant case law and legal precedents",
        weight=1.0,
        scoring_guide="1=Missing precedents, 3=Some precedents cited, 5=Comprehensive precedent analysis",
    )

    # Communication Category
    communication = builder.add_category(
        name="Legal Communication",
        description="Clarity and persuasiveness of legal communication",
        weight=1.5,
    )
    communication.add_criterion(
        name="Clarity of Argument",
        description="Legal arguments presented clearly and accessibly",
        weight=1.0,
        scoring_guide="1=Confusing arguments, 3=Generally clear, 5=Exceptionally clear legal communication",
    )
    communication.add_criterion(
        name="Professional Demeanor",
        description="Maintains appropriate professional tone and courtroom presence",
        weight=0.5,
        scoring_guide="1=Unprofessional, 3=Professional, 5=Highly authoritative presence",
    )

    builder.set_scoring_scale(
        min_score=1,
        max_score=100,
        labels={
            1: "Poor - Legally inadequate",
            20: "Below Average - Significant legal issues",
            40: "Average - Basic legal competence",
            60: "Good - Strong legal presentation",
            80: "Very Good - Compelling legal arguments",
            100: "Excellent - Outstanding legal advocacy",
        },
    )

    return builder.build(
        is_template=True, tags=["legal", "court", "law", "advocacy", "argument"]
    )


def create_medical_presentation_rubric() -> Rubric:
    """Create a rubric for medical and healthcare presentations."""
    builder = RubricBuilder(
        name="Medical & Healthcare Presentation",
        description="Evaluate medical research, patient education, and healthcare presentations",
    )

    # Medical Knowledge Category
    medical_knowledge = builder.add_category(
        name="Medical Knowledge & Accuracy",
        description="Accuracy and depth of medical information presented",
        weight=2.5,
    )
    medical_knowledge.add_criterion(
        name="Medical Accuracy",
        description="Information is medically accurate and up-to-date",
        weight=1.5,
        scoring_guide="1=Medical errors present, 3=Generally accurate, 5=Completely accurate and authoritative",
    )
    medical_knowledge.add_criterion(
        name="Evidence-Based Practice",
        description="Presentation grounded in current medical evidence and research",
        weight=1.0,
        scoring_guide="1=Not evidence-based, 3=Some evidence cited, 5=Strong evidence-based approach",
    )

    # Patient/Customer Focus Category
    patient_focus = builder.add_category(
        name="Patient/Customer Focus",
        description="Effectiveness in addressing patient/customer needs",
        weight=2.0,
    )
    patient_focus.add_criterion(
        name="Patient Education",
        description="Effectively educates patients/customers about medical conditions/treatments",
        weight=1.0,
        scoring_guide="1=Poor patient education, 3=Adequate information, 5=Excellent patient education",
    )
    patient_focus.add_criterion(
        name="Empathy & Communication",
        description="Demonstrates empathy and clear communication with patients/customers",
        weight=1.0,
        scoring_guide="1=Lacks empathy, 3=Some empathy shown, 5=Highly empathetic and clear",
    )

    # Ethical & Professional Category
    ethics = builder.add_category(
        name="Ethics & Professionalism",
        description="Ethical considerations and professional standards",
        weight=1.0,
    )
    ethics.add_criterion(
        name="Ethical Considerations",
        description="Addresses ethical implications and patient rights appropriately",
        weight=0.5,
        scoring_guide="1=Ignores ethics, 3=Mentions ethics, 5=Comprehensive ethical analysis",
    )
    ethics.add_criterion(
        name="Professional Standards",
        description="Maintains high professional standards and patient confidentiality",
        weight=0.5,
        scoring_guide="1=Unprofessional, 3=Professional, 5=Exemplary professionalism",
    )

    # Practical Application Category
    application = builder.add_category(
        name="Practical Application",
        description="Practical relevance and implementation",
        weight=0.5,
    )
    application.add_criterion(
        name="Clinical Relevance",
        description="Information is clinically relevant and applicable",
        weight=0.5,
        scoring_guide="1=Not clinically relevant, 3=Some relevance, 5=Highly clinically relevant",
    )

    builder.set_scoring_scale(
        min_score=1,
        max_score=100,
        labels={
            1: "Poor - Medically inadequate",
            20: "Below Average - Significant medical issues",
            40: "Average - Basic medical competence",
            60: "Good - Strong medical presentation",
            80: "Very Good - Excellent medical communication",
            100: "Excellent - Outstanding medical communication",
        },
    )

    return builder.build(
        is_template=True,
        tags=["medical", "healthcare", "patient", "clinical", "education"],
    )


def create_political_social_rubric() -> Rubric:
    """Create a rubric for political and social presentations."""
    builder = RubricBuilder(
        name="Political & Social Presentation",
        description="Evaluate political speeches, campaign presentations, and social advocacy",
    )

    # Message & Vision Category
    message = builder.add_category(
        name="Message & Vision",
        description="Clarity and persuasiveness of core message and vision",
        weight=2.0,
    )
    message.add_criterion(
        name="Clear Message",
        description="Core message and position clearly articulated",
        weight=1.0,
        scoring_guide="1=Unclear message, 3=Generally clear, 5=Crystal clear and compelling message",
    )
    message.add_criterion(
        name="Vision & Goals",
        description="Presents clear vision and achievable goals",
        weight=1.0,
        scoring_guide="1=Vague goals, 3=Some goals stated, 5=Inspiring vision with clear goals",
    )

    # Evidence & Facts Category
    evidence = builder.add_category(
        name="Evidence & Factual Support",
        description="Use of facts, data, and evidence to support positions",
        weight=2.0,
    )
    evidence.add_criterion(
        name="Factual Accuracy",
        description="Information presented is factually accurate and verifiable",
        weight=1.0,
        scoring_guide="1=Factual errors, 3=Mostly accurate, 5=Completely factually accurate",
    )
    evidence.add_criterion(
        name="Data & Research",
        description="Appropriate use of data, research, and supporting evidence",
        weight=1.0,
        scoring_guide="1=No supporting data, 3=Some data cited, 5=Comprehensive data-driven arguments",
    )

    # Emotional Appeal & Connection Category
    emotional = builder.add_category(
        name="Emotional Appeal & Connection",
        description="Ability to connect emotionally and inspire action",
        weight=1.5,
    )
    emotional.add_criterion(
        name="Emotional Connection",
        description="Creates emotional connection with audience through storytelling",
        weight=1.0,
        scoring_guide="1=No emotional appeal, 3=Some emotional elements, 5=Powerful emotional connection",
    )
    emotional.add_criterion(
        name="Call to Action",
        description="Clear, motivating call to action for audience engagement",
        weight=0.5,
        scoring_guide="1=No clear action, 3=Mentioned action, 5=Compelling, actionable call",
    )

    # Inclusivity & Ethics Category
    inclusivity = builder.add_category(
        name="Inclusivity & Ethics",
        description="Addresses diverse perspectives and maintains ethical standards",
        weight=0.5,
    )
    inclusivity.add_criterion(
        name="Inclusive Messaging",
        description="Considers and addresses diverse audience perspectives",
        weight=0.5,
        scoring_guide="1=Excludes perspectives, 3=Some inclusivity, 5=Highly inclusive messaging",
    )

    builder.set_scoring_scale(
        min_score=1,
        max_score=100,
        labels={
            1: "Poor - Not persuasive",
            20: "Below Average - Lacks impact",
            40: "Average - Basic effectiveness",
            60: "Good - Strong persuasive presentation",
            80: "Very Good - Compelling advocacy",
            100: "Excellent - Outstanding advocacy and persuasion",
        },
    )

    return builder.build(
        is_template=True, tags=["political", "social", "campaign", "advocacy", "speech"]
    )


def create_entertainment_media_rubric() -> Rubric:
    """Create a rubric for entertainment and media presentations."""
    builder = RubricBuilder(
        name="Entertainment & Media Presentation",
        description="Evaluate entertainment pitches, media proposals, and creative media presentations",
    )

    # Creative Concept Category
    concept = builder.add_category(
        name="Creative Concept & Vision",
        description="Originality and appeal of the creative concept",
        weight=2.5,
    )
    concept.add_criterion(
        name="Originality & Innovation",
        description="Concept demonstrates creative originality and market differentiation",
        weight=1.5,
        scoring_guide="1=Derivative concept, 3=Some originality, 5=Highly innovative and unique",
    )
    concept.add_criterion(
        name="Market Appeal",
        description="Concept has clear appeal to target audience and market potential",
        weight=1.0,
        scoring_guide="1=Limited appeal, 3=Some market potential, 5=Strong market appeal and potential",
    )

    # Storytelling & Content Category
    storytelling = builder.add_category(
        name="Storytelling & Content Quality",
        description="Quality of narrative and content development",
        weight=2.0,
    )
    storytelling.add_criterion(
        name="Narrative Quality",
        description="Compelling story structure and character development",
        weight=1.0,
        scoring_guide="1=Weak narrative, 3=Adequate story, 5=Outstanding storytelling",
    )
    storytelling.add_criterion(
        name="Content Appropriateness",
        description="Content appropriate for target audience and platform",
        weight=1.0,
        scoring_guide="1=Inappropriate content, 3=Generally appropriate, 5=Perfectly targeted content",
    )

    # Production & Execution Category
    production = builder.add_category(
        name="Production & Execution",
        description="Quality of production values and execution",
        weight=1.5,
    )
    production.add_criterion(
        name="Production Quality",
        description="Technical and production quality meets industry standards",
        weight=1.0,
        scoring_guide="1=Poor production, 3=Adequate quality, 5=Professional production values",
    )
    production.add_criterion(
        name="Feasibility",
        description="Concept is realistically feasible within budget and timeline",
        weight=0.5,
        scoring_guide="1=Not feasible, 3=Challenging but possible, 5=Highly feasible",
    )

    builder.set_scoring_scale(
        min_score=1,
        max_score=100,
        labels={
            1: "Poor - Not marketable",
            20: "Below Average - Needs significant development",
            40: "Average - Basic entertainment value",
            60: "Good - Strong entertainment potential",
            80: "Very Good - Promising creative work",
            100: "Excellent - Outstanding creative and commercial potential",
        },
    )

    return builder.build(
        is_template=True,
        tags=["entertainment", "media", "pitch", "creative", "production"],
    )


def create_creative_presentation_rubric() -> Rubric:
    """Create a rubric for creative and artistic presentations."""
    builder = RubricBuilder(
        name="Creative Presentation",
        description="Evaluate creative work, artistic portfolios, and innovative project presentations",
    )

    # Creative Vision & Originality Category
    creative = builder.add_category(
        name="Creative Vision & Originality",
        description="Originality and artistic vision of the creative work",
        weight=2.5,
    )
    creative.add_criterion(
        name="Originality & Innovation",
        description="Work demonstrates unique creativity and innovative approaches",
        weight=1.5,
        scoring_guide="1=Derivative/unoriginal, 3=Some originality, 5=Highly innovative and unique",
    )
    creative.add_criterion(
        name="Artistic Vision",
        description="Clear artistic intent and conceptual framework",
        weight=1.0,
        scoring_guide="1=Unclear vision, 3=Developing vision, 5=Compelling, well-articulated vision",
    )
    creative.add_criterion(
        name="Creative Process",
        description="Effective explanation of the creative process and decision-making",
        weight=1.0,
        scoring_guide="1=Process unclear, 3=Some process insight, 5=Excellent process documentation",
    )

    # Aesthetic & Technical Quality Category
    aesthetic = builder.add_category(
        name="Aesthetic & Technical Quality",
        description="Quality of the creative output and technical execution",
        weight=2.0,
    )
    aesthetic.add_criterion(
        name="Aesthetic Quality",
        description="Visual/design appeal and artistic craftsmanship",
        weight=1.0,
        scoring_guide="1=Poor aesthetics, 3=Adequate quality, 5=Outstanding aesthetic achievement",
    )
    aesthetic.add_criterion(
        name="Technical Proficiency",
        description="Mastery of tools, techniques, and medium-specific skills",
        weight=1.0,
        scoring_guide="1=Technical weaknesses, 3=Competent execution, 5=Masterful technical skill",
    )
    aesthetic.add_criterion(
        name="Portfolio Presentation",
        description="Effective presentation and curation of creative work",
        weight=0.5,
        scoring_guide="1=Poor presentation, 3=Adequate display, 5=Professional, compelling presentation",
    )

    # Impact & Communication Category
    impact = builder.add_category(
        name="Impact & Communication",
        description="Emotional impact and ability to communicate creative intent",
        weight=1.5,
    )
    impact.add_criterion(
        name="Emotional Resonance",
        description="Work evokes emotional response and engages audience",
        weight=1.0,
        scoring_guide="1=No emotional impact, 3=Some engagement, 5=Powerful emotional connection",
    )
    impact.add_criterion(
        name="Concept Communication",
        description="Successfully communicates creative concepts and ideas",
        weight=0.75,
        scoring_guide="1=Ideas unclear, 3=Concepts conveyed, 5=Brilliant concept communication",
    )

    builder.set_scoring_scale(
        min_score=1,
        max_score=100,
        labels={
            1: "Poor - Lacks creative merit",
            20: "Below Average - Developing talent",
            40: "Average - Solid creative work",
            60: "Good - Strong creative presentation",
            80: "Very Good - Impressive creative work",
            100: "Excellent - Outstanding creative achievement",
        },
    )

    return builder.build(
        is_template=True, tags=["creative", "artistic", "portfolio", "design"]
    )


def create_general_presentation_rubric() -> Rubric:
    """Create a general-purpose rubric for any type of presentation."""
    builder = RubricBuilder(
        name="General Presentation",
        description="Evaluate any type of presentation on core presentation skills",
    )

    # Content Category
    content = builder.add_category(
        name="Content",
        description="Quality and relevance of presentation content",
        weight=1.5,
    )
    content.add_criterion(
        name="Relevance & Focus",
        description="Content is relevant and focused on the topic",
        weight=1.0,
        scoring_guide="1=Off-topic, 3=Generally relevant, 5=Highly relevant and focused",
    )
    content.add_criterion(
        name="Completeness",
        description="Covers the material comprehensively",
        weight=1.0,
        scoring_guide="1=Incomplete, 3=Adequate coverage, 5=Thorough coverage",
    )
    content.add_criterion(
        name="Accuracy",
        description="Information presented is accurate",
        weight=1.0,
        scoring_guide="1=Multiple errors, 3=Mostly accurate, 5=Completely accurate",
    )

    # Delivery Category
    delivery = builder.add_category(
        name="Delivery",
        description="Speaking skills and presence",
        weight=1.5,
    )
    delivery.add_criterion(
        name="Clarity",
        description="Spoken clearly and at appropriate pace",
        weight=1.0,
        scoring_guide="1=Hard to understand, 3=Generally clear, 5=Very clear",
    )
    delivery.add_criterion(
        name="Confidence",
        description="Presents with poise and confidence",
        weight=0.75,
        scoring_guide="1=Very nervous, 3=Reasonably confident, 5=Very confident",
    )
    delivery.add_criterion(
        name="Engagement",
        description="Engages and maintains audience attention",
        weight=0.75,
        scoring_guide="1=Boring, 3=Moderately engaging, 5=Highly engaging",
    )

    # Organization Category
    org = builder.add_category(
        name="Organization",
        description="Structure and logical flow",
        weight=1.0,
    )
    org.add_criterion(
        name="Logical Flow",
        description="Ideas presented in logical sequence",
        weight=1.0,
        scoring_guide="1=Disorganized, 3=Generally logical, 5=Excellent flow",
    )
    org.add_criterion(
        name="Transitions",
        description="Smooth transitions between ideas and sections",
        weight=0.5,
        scoring_guide="1=Abrupt transitions, 3=Adequate transitions, 5=Smooth transitions",
    )

    # Visual Category
    visual = builder.add_category(
        name="Visuals", description="Quality of visual aids", weight=1.0
    )
    visual.add_criterion(
        name="Visual Quality",
        description="Slides/visuals are professional and clear",
        weight=1.0,
        scoring_guide="1=Poor quality, 3=Adequate, 5=Professional",
    )

    builder.set_scoring_scale(
        min_score=1,
        max_score=100,
        labels={
            1: "Poor",
            20: "Below Average",
            40: "Average",
            60: "Good",
            80: "Very Good",
            100: "Excellent",
        },
    )

    return builder.build(is_template=True, tags=["general", "presentation"])


def get_default_rubric(rubric_type: str) -> Rubric | None:
    """Get a default rubric by type.

    Args:
        rubric_type: Type of rubric (academic, business, teaching, technical, creative, sales, legal, medical, political, entertainment, general)

    Returns:
        Rubric object or None if type not found
    """
    rubrics = {
        "academic": create_academic_presentation_rubric,
        "business": create_business_pitch_rubric,
        "teaching": create_teaching_demo_rubric,
        "technical": create_technical_presentation_rubric,
        "creative": create_creative_presentation_rubric,
        "sales": create_sales_marketing_rubric,
        "legal": create_legal_presentation_rubric,
        "medical": create_medical_presentation_rubric,
        "political": create_political_social_rubric,
        "entertainment": create_entertainment_media_rubric,
        "general": create_general_presentation_rubric,
    }

    builder = rubrics.get(rubric_type.lower())
    return builder() if builder else None


def list_default_rubrics() -> list[str]:
    """List all available default rubric types."""
    return [
        "academic",
        "business",
        "teaching",
        "technical",
        "creative",
        "sales",
        "legal",
        "medical",
        "political",
        "entertainment",
        "general",
    ]
