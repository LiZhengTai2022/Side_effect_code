# -*- coding: utf-8 -*-
import requests
import json
import pandas as pd
import time
import re
import os

# ==================== 配置区域 ====================
# 在这里设置您的API密钥，如果不想每次都手动输入的话

# PubMed API密钥（可选）
# 获取地址：https://www.ncbi.nlm.nih.gov/account/
PUBMED_API_KEY = ""  # 请在此处填入您的PubMed API密钥，如无则留空

# DeepSeek API密钥（可选，用于AI分析）
# 获取地址：https://platform.deepseek.com/
DEEPSEEK_API_KEY = ""  # 请在此处填入您的DeepSeek API密钥，如无则留空

# 默认文献分析数量
DEFAULT_MAX_RESULTS = 10000

# 默认输出文件名
DEFAULT_OUTPUT_FILE = "药物副作用_菌群组学文献分析.xlsx"

# ==================== 配置区域结束 ====================

class PubMedLiteratureAnalyzer:
    def __init__(self, api_key=None, ai_api_key=None):
        self.api_key = api_key
        self.ai_api_key = ai_api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.results = []
        
    def search_pubmed(self, keywords, max_results=20):
        # 如果关键词已经是格式化的PubMed查询，直接使用；否则进行简单处理
        if len(keywords) == 1 and ('(' in keywords[0] or '[' in keywords[0]):
            # 已经是格式化的查询，直接使用
            search_query = keywords[0]
        else:
            # 简单的关键词，进行基本处理
            search_query = " AND ".join(['"' + kw + '"' for kw in keywords])
        
        # 使用POST方法以避免URL长度限制
        search_url = "{}esearch.fcgi".format(self.base_url)
        
        # 构建POST参数
        params = {
            'db': 'pubmed',
            'term': search_query,
            'retmax': str(max_results),
            'retmode': 'json'
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            response = requests.post(search_url, data=params)
            response.raise_for_status()
            data = response.json()
            pmids = data.get('esearchresult', {}).get('idlist', [])
            
            if not pmids:
                print("未找到相关文献")
                return []
            
            print("找到 {} 篇相关文献".format(len(pmids)))
            return self.fetch_abstracts(pmids)
            
        except requests.RequestException as e:
            print("搜索PubMed时出错: {}".format(e))
            return []
    
    def fetch_abstracts(self, pmids):
        papers = []
        batch_size = 100
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i+batch_size]
            pmid_string = ",".join(batch_pmids)
            
            # 使用POST方法以避免URL长度限制
            fetch_url = "{}efetch.fcgi".format(self.base_url)
            
            # 构建POST参数
            params = {
                'db': 'pubmed',
                'id': pmid_string,
                'retmode': 'xml'
            }
            
            if self.api_key:
                params['api_key'] = self.api_key
            
            try:
                response = requests.post(fetch_url, data=params)
                response.raise_for_status()
                papers.extend(self.parse_pubmed_xml(response.text))
                time.sleep(0.34)  # 遵守PubMed API速率限制
                
            except requests.RequestException as e:
                print("获取摘要时出错: {}".format(e))
                continue
        
        return papers
    
    def parse_pubmed_xml(self, xml_content):
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_content)
        papers = []
        
        for article in root.findall('.//PubmedArticle'):
            try:
                pmid = article.find('.//PMID').text
                title = article.find('.//ArticleTitle').text or ""
                
                abstract_elem = article.find('.//Abstract/AbstractText')
                abstract = ""
                if abstract_elem is not None:
                    abstract = abstract_elem.text or ""
                    for label in abstract_elem.findall('.//AbstractText'):
                        if label.text:
                            abstract += " " + label.text
                
                authors = []
                author_list = article.find('.//AuthorList')
                if author_list is not None:
                    for author in author_list.findall('.//Author'):
                        last_name = author.find('.//LastName')
                        first_name = author.find('.//ForeName')
                        if last_name is not None and first_name is not None:
                            authors.append("{} {}".format(first_name.text, last_name.text))
                
                journal = article.find('.//Journal/Title')
                journal_name = journal.text if journal is not None else ""
                
                pub_date = article.find('.//PubDate/Year')
                year = pub_date.text if pub_date is not None else ""
                
                paper = {
                    'pmid': pmid,
                    'title': title,
                    'abstract': abstract,
                    'authors': ', '.join(authors),
                    'journal': journal_name,
                    'year': year,
                    'url': "https://pubmed.ncbi.nlm.nih.gov/{}/".format(pmid)
                }
                papers.append(paper)
                
            except Exception as e:
                print("解析文献 {} 时出错: {}".format(pmid if 'pmid' in locals() else 'unknown', e))
                continue
        
        return papers
    
    def analyze_with_ai(self, paper):
        if not self.ai_api_key:
            print("未提供AI API密钥，跳过AI分析")
            return self.basic_analysis(paper)
        
        prompt = """
        请分析以下医学文献，提取以下信息：
        1. 药物名称和分子结构信息
        2. 副作用类型（特别关注以下25大类副作用）：
           - 胃肠道：腹泻、结肠炎、胃肠道毒性
           - 肝脏：肝毒性、药物性肝损伤
           - 心脏：心脏毒性、药物性心脏毒性、长QT综合征
           - 肾脏：肾毒性、急性肾损伤
           - 代谢/内分泌：高血糖、代谢副作用、体重增加、2型糖尿病
           - 神经/精神：神经毒性、化疗诱导的周围神经病变
           - 血液：骨髓抑制、中性粒细胞减少、血小板减少、贫血
           - 皮肤：皮肤毒性、皮疹、手足综合征、脱发
           - 免疫相关不良反应：免疫相关不良事件、irAE、免疫检查点抑制剂毒性、细胞因子释放综合征
           - 呼吸/肺部：肺毒性、间质性肺病、肺纤维化
           - 骨骼肌肉：肌痛、关节痛、横纹肌溶解、骨毒性、骨质疏松
           - 眼：眼毒性、视网膜病变、视力障碍
           - 耳：耳毒性、听力损失、耳鸣
           - 生殖泌尿：生殖毒性、不孕不育、性功能障碍、尿潴留、月经紊乱
           - 血管/循环：高血压、血栓形成、水肿、出血、血管毒性
           - 疲劳/衰弱：疲劳、衰弱
           - 过敏/超敏反应：超敏反应、过敏性休克
           - 口腔黏膜/唾液腺：口腔黏膜炎、口干症
           - 味觉/嗅觉：味觉障碍、嗅觉障碍
           - 药物诱发的感染：药物诱发感染、艰难梭菌感染、机会性感染
           - 胰腺毒性：药物诱发胰腺炎、胰腺炎
           - 甲状腺功能异常：甲状腺功能障碍、甲状腺炎
           - 电解质紊乱：电解质失衡、低钾血症
           - 认知障碍/脑雾：认知障碍、脑雾
           - 性腺/生殖内分泌紊乱：性腺毒性、性腺功能减退、多毛症
        3. 菌群组学数据的获取方法或下载地址（16S rRNA测序、宏基因组测序、微生物组测序等）
        4. 数据可用性声明（Data Availability Statement）：寻找文献中关于数据可用性的声明部分，如"Data Availability"、"Data Accessibility"、"Data Sharing"等部分，将其归纳为一句话
        5. 菌群分析详情：从摘要和结果部分深入分析，提取以下信息：
           - 主要菌群类型（如：厚壁菌门、拟杆菌门、变形菌门、放线菌门等）
           - 具体菌种变化（如：Bacteroides fragilis、Faecalibacterium prausnitzii、Escherichia coli等）
           - 相关指标（如：相对丰度、多样性指数、菌群失调程度、特定菌属比例变化等）
           - 菌群变化趋势（增加/减少/显著变化）
           - 与药物副作用的关系
        
        文献标题: {}
        文献摘要: {}
        
        请以JSON格式返回结果，格式如下：
        {{
            "drug_name": "药物名称",
            "drug_structure": "药物分子结构描述",
            "side_effects": ["副作用类型1", "副作用类型2"],
            "microbiome_data_method": "微生物组数据获取方法（如16S rRNA测序、宏基因组测序等）",
            "data_download_url": "数据下载地址（如GEO、SRA、EBI等数据库链接）",
            "data_availability_statement": "数据可用性声明（一句话归纳，如无则为null）",
            "microbiome_analysis": {{
                "main_phyla": ["主要菌群门类1", "主要菌群门类2"],
                "specific_species": [
                    {{
                        "species": "菌种名称",
                        "change": "增加/减少/显著变化",
                        "indicators": "相关指标（如相对丰度变化、比例变化等）",
                        "correlation": "与副作用的关系"
                    }}
                ],
                "diversity_metrics": "多样性指标变化",
                "overall_pattern": "整体菌群变化模式"
            }},
            "confidence_score": 0.8
        }}
        
        如果某些信息在文献中没有找到，请设置为null。
        """.format(paper['title'], paper['abstract'])
        
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": "Bearer {}".format(self.ai_api_key),
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "你是一个专业的医学文献分析专家，擅长从医学文献中提取药物、副作用和微生物组学数据信息。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1000
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            content = result['choices'][0]['message']['content']
            
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                analysis = json.loads(json_match.group())
                return analysis
            else:
                print("AI返回格式异常: {}".format(content))
                return self.basic_analysis(paper)
                
        except Exception as e:
            print("AI分析时出错: {}".format(e))
            return self.basic_analysis(paper)
    
    def basic_analysis(self, paper):
        abstract = paper['abstract'].lower()
        title = paper['title'].lower()
        
        drug_keywords = ['drug', 'medication', 'pharmaceutical', 'treatment', 'therapy']
        side_effect_keywords = [
            # 胃肠道
            'diarrhea', 'colitis', 'gastrointestinal toxicity',
            # 肝脏
            'hepatotoxicity', 'drug-induced liver injury', 'liver injury',
            # 心脏
            'cardiotoxicity', 'drug-induced cardiac toxicity', 'cardiac toxicity',
            # 肾脏
            'nephrotoxicity', 'acute kidney injury', 'kidney injury', 'renal toxicity',
            # 代谢/内分泌
            'hyperglycemia', 'metabolic side effect', 'weight gain', 'diabetes',
            # 神经/精神
            'neurotoxicity', 'chemotherapy-induced peripheral neuropathy', 'neuropathy',
            # 血液
            'myelosuppression', 'neutropenia', 'thrombocytopenia', 'anemia',
            # 皮肤
            'skin toxicity', 'rash', 'hand-foot syndrome', 'alopecia',
            # 免疫相关
            'immune-related adverse event', 'irae', 'immune checkpoint inhibitor toxicity', 'cytokine release',
            # 呼吸/肺部
            'pulmonary toxicity', 'interstitial lung disease', 'pulmonary fibrosis', 'lung toxicity',
            # 骨骼肌肉
            'myalgia', 'arthralgia', 'rhabdomyolysis', 'bone toxicity', 'osteoporosis',
            # 眼
            'ocular toxicity', 'retinopathy', 'vision disorder',
            # 耳
            'ototoxicity', 'hearing loss', 'tinnitus',
            # 生殖泌尿
            'reproductive toxicity', 'infertility', 'sexual dysfunction', 'urinary retention', 'menstrual disorder',
            # 血管/循环
            'hypertension', 'thrombosis', 'edema', 'bleeding', 'vascular toxicity',
            # 疲劳/衰弱
            'fatigue', 'asthenia',
            # 过敏/超敏反应
            'hypersensitivity', 'anaphylaxis', 'allergic reaction',
            # 口腔黏膜/唾液腺
            'oral mucositis', 'xerostomia', 'stomatitis',
            # 味觉/嗅觉
            'dysgeusia', 'anosmia', 'taste disorder', 'smell disorder',
            # 药物诱发的感染
            'drug-induced infection', 'clostridium difficile', 'opportunistic infection',
            # 胰腺毒性
            'drug-induced pancreatitis', 'pancreatitis',
            # 甲状腺功能异常
            'thyroid dysfunction', 'thyroiditis',
            # 电解质紊乱
            'electrolyte imbalance', 'hypokalemia',
            # 认知障碍/脑雾
            'cognitive impairment', 'brain fog',
            # 性腺/生殖内分泌紊乱
            'gonadal toxicity', 'hypogonadism', 'hirsutism',
            # 兜底词
            'side effect', 'adverse effect', 'toxicity', 'adverse reaction'
        ]
        microbiome_keywords = ['microbiome', 'microbiota', 'gut flora', '16s', 'metagenomics', 'sequencing']
        
        drug_name = None
        for keyword in drug_keywords:
            if keyword in abstract or keyword in title:
                sentences = re.split(r'[.!?]', abstract + ' ' + title)
                for sentence in sentences:
                    if keyword in sentence.lower():
                        words = sentence.split()
                        for i, word in enumerate(words):
                            if keyword in word.lower() and i + 1 < len(words):
                                potential_drug = words[i + 1].strip('.,;:()[]{}"\'')
                                if len(potential_drug) > 2:
                                    drug_name = potential_drug
                                    break
                        if drug_name:
                            break
                if drug_name:
                    break
        
        side_effects = []
        for keyword in side_effect_keywords:
            if keyword in abstract:
                sentences = re.split(r'[.!?]', abstract)
                for sentence in sentences:
                    if keyword in sentence.lower():
                        side_effects.append(sentence.strip()[:100])
        
        microbiome_method = None
        for keyword in microbiome_keywords:
            if keyword in abstract:
                sentences = re.split(r'[.!?]', abstract)
                for sentence in sentences:
                    if keyword in sentence.lower():
                        microbiome_method = sentence.strip()[:200]
                        break
                if microbiome_method:
                    break
        
        # 基础菌群分析提取
        microbiome_analysis = self.basic_microbiome_analysis(abstract)
        
        return {
            'drug_name': drug_name,
            'drug_structure': None,
            'side_effects': side_effects[:3] if side_effects else [],
            'microbiome_data_method': microbiome_method,
            'data_download_url': None,
            'data_availability_statement': None,
            'microbiome_analysis': microbiome_analysis,
            'confidence_score': 0.3
        }
    
    def basic_microbiome_analysis(self, text):
        """
        基础菌群分析方法，从文本中提取菌群相关信息
        """
        # 常见菌群门类
        phyla_keywords = [
            'firmicutes', 'bacteroidetes', 'proteobacteria', 'actinobacteria',
            'verrucomicrobia', 'fusobacteria', 'cyanobacteria', 'spirits',
            '厚壁菌门', '拟杆菌门', '变形菌门', '放线菌门'
        ]
        
        # 常见菌属
        genus_keywords = [
            'bacteroides', 'prevotella', 'faecalibacterium', 'lactobacillus',
            'bifidobacterium', 'escherichia', 'clostridium', 'streptococcus',
            'enterococcus', 'staphylococcus', 'akermansia', 'roseburia',
            '拟杆菌属', '普雷沃菌属', '粪杆菌属', '乳杆菌属', '双歧杆菌属'
        ]
        
        # 常见菌种
        species_keywords = [
            'bacteroides fragilis', 'bacteroides thetaiotaomicron',
            'faecalibacterium prausnitzii', 'escherichia coli',
            'lactobacillus acidophilus', 'bifidobacterium longum',
            'akermansia muciniphila', 'clostridium difficile'
        ]
        
        # 多样性指标关键词
        diversity_keywords = [
            'diversity', 'shannon', 'simpson', 'chao1', 'ace',
            'richness', 'evenness', 'alpha diversity', 'beta diversity',
            '多样性', '丰富度', '均匀度'
        ]
        
        # 变化趋势关键词
        change_keywords = [
            'increased', 'decreased', 'enriched', 'reduced',
            'significant', 'correlation', 'associated',
            '增加', '减少', '显著', '相关', '丰度'
        ]
        
        # 提取主要菌群门类
        main_phyla = []
        for phylum in phyla_keywords:
            if phylum.lower() in text.lower():
                main_phyla.append(phylum)
        
        # 提取具体菌种信息
        specific_species = []
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            for species in species_keywords:
                if species.lower() in sentence.lower():
                    # 提取变化趋势和指标
                    change = "未知"
                    for change_word in change_keywords:
                        if change_word.lower() in sentence.lower():
                            if change_word in ['increased', 'enriched', '增加']:
                                change = "增加"
                            elif change_word in ['decreased', 'reduced', '减少']:
                                change = "减少"
                            elif change_word in ['significant', '显著']:
                                change = "显著变化"
                            break
                    
                    # 提取相关指标
                    indicators = []
                    for div_keyword in diversity_keywords:
                        if div_keyword.lower() in sentence.lower():
                            indicators.append(div_keyword)
                    
                    specific_species.append({
                        'species': species,
                        'change': change,
                        'indicators': '; '.join(indicators) if indicators else '未明确指标',
                        'correlation': sentence.strip()[:150]
                    })
                    break
        
        # 提取多样性指标
        diversity_metrics = []
        for div_keyword in diversity_keywords:
            if div_keyword.lower() in text.lower():
                # 找到包含多样性指标的句子
                for sentence in sentences:
                    if div_keyword.lower() in sentence.lower():
                        diversity_metrics.append(sentence.strip()[:100])
                        break
        
        # 整体模式
        overall_pattern = None
        pattern_keywords = ['dysbiosis', 'imbalance', 'shift', 'alteration', '失调', '失衡', '变化']
        for sentence in sentences:
            for pattern_word in pattern_keywords:
                if pattern_word.lower() in sentence.lower():
                    overall_pattern = sentence.strip()[:200]
                    break
            if overall_pattern:
                break
        
        return {
            'main_phyla': main_phyla[:5] if main_phyla else [],
            'specific_species': specific_species[:10] if specific_species else [],
            'diversity_metrics': '; '.join(diversity_metrics[:3]) if diversity_metrics else None,
            'overall_pattern': overall_pattern
        }
    
    def extract_data_repository_info(self, paper):
        abstract = paper['abstract'] + ' ' + paper['title']
        repositories = {
            'GEO': r'geo\s*series\s*accession\s*[:\s]*(GSE\d+)',
            'SRA': r'sra\s*accession\s*[:\s]*(SR[AP]\d+)',
            'EBI': r'ebi\s*accession\s*[:\s]*(ERP\d+|SRP\d+)',
            'ArrayExpress': r'arrayexpress\s*[:\s]*(E-\w+-\d+)',
            'GitHub': r'github\.com/[\w\-]+/[\w\-]+',
            'Zenodo': r'zenodo\.org/record/\d+',
            'Figshare': r'figshare\.com/articles/\d+',
            'Dryad': r'datadryad\.org/stash/dataset/doi:[\w\./\-]+'
        }
        
        found_repos = {}
        for repo_name, pattern in repositories.items():
            matches = re.findall(pattern, abstract, re.IGNORECASE)
            if matches:
                found_repos[repo_name] = matches
        
        return found_repos
    
    def extract_data_availability_statement(self, paper):
        """
        提取文献中的数据可用性声明
        寻找类似"Data Availability"、"Data Accessibility"、"Data Sharing"等部分
        """
        abstract = paper['abstract'] + ' ' + paper['title']
        
        # 数据可用性声明的常见关键词和模式
        availability_keywords = [
            'data availability',
            'data accessibility', 
            'data sharing',
            'availability of data',
            'data statement',
            'ethical statement',
            'data deposition',
            'publicly available',
            'available upon request',
            'supplementary material',
            'supplementary data',
            'additional data',
            'raw data',
            'sequencing data',
            'microbiome data',
            '16s data'
        ]
        
        # 寻找包含数据可用性关键词的句子
        availability_sentences = []
        sentences = re.split(r'[.!?]', abstract)
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            for keyword in availability_keywords:
                if keyword in sentence_lower:
                    # 清理和格式化句子
                    cleaned_sentence = ' '.join(sentence.split())
                    if len(cleaned_sentence) > 20:  # 过滤掉太短的片段
                        availability_sentences.append(cleaned_sentence)
                    break
        
        # 如果找到了相关句子，尝试归纳为一句话
        if availability_sentences:
            # 取第一个最相关的句子
            primary_statement = availability_sentences[0]
            
            # 如果句子太长，尝试截取关键部分
            if len(primary_statement) > 300:
                # 寻找关键短语的位置
                key_phrases = ['data', 'available', 'access', 'download', 'repository', 'database', 'supplementary']
                for phrase in key_phrases:
                    if phrase in primary_statement.lower():
                        # 找到包含关键短语的部分
                        phrase_pos = primary_statement.lower().find(phrase)
                        start = max(0, phrase_pos - 50)
                        end = min(len(primary_statement), phrase_pos + 150)
                        truncated = primary_statement[start:end].strip()
                        if len(truncated) > 30:
                            return truncated
            
            return primary_statement[:300]  # 限制最大长度
        
        return None
    
    def process_papers(self, keywords, max_results=20):
        print("开始搜索文献，关键词: {}".format(keywords))
        papers = self.search_pubmed(keywords, max_results)
        
        if not papers:
            print("没有找到文献")
            return []
        
        print("开始分析 {} 篇文献...".format(len(papers)))
        analyzed_results = []
        
        for i, paper in enumerate(papers, 1):
            print("正在分析第 {}/{} 篇文献: {}...".format(i, len(papers), paper['title'][:50]))
            
            analysis = self.analyze_with_ai(paper)
            repo_info = self.extract_data_repository_info(paper)
            availability_statement = self.extract_data_availability_statement(paper)
            
            # 确保side_effects是列表格式
            side_effects = analysis.get('side_effects', [])
            if not isinstance(side_effects, list):
                if side_effects is None:
                    side_effects = []
                elif isinstance(side_effects, str):
                    side_effects = [side_effects]
                else:
                    side_effects = list(side_effects) if hasattr(side_effects, '__iter__') else []
            
            # 如果AI没有提取到数据可用性声明，使用基础方法提取
            data_availability = analysis.get('data_availability_statement') or availability_statement
            
            # 处理菌群分析信息
            microbiome_analysis = analysis.get('microbiome_analysis') or self.basic_microbiome_analysis(paper['abstract'])
            
            # 格式化主要菌群门类
            main_phyla = microbiome_analysis.get('main_phyla', [])
            if isinstance(main_phyla, list):
                main_phyla_str = '; '.join(main_phyla)
            else:
                main_phyla_str = str(main_phyla) if main_phyla else ''
            
            # 格式化具体菌种信息
            specific_species = microbiome_analysis.get('specific_species', [])
            species_details = []
            if isinstance(specific_species, list):
                for species_info in specific_species[:5]:  # 限制显示前5个菌种
                    if isinstance(species_info, dict):
                        species_name = species_info.get('species', '未知菌种')
                        change = species_info.get('change', '未知')
                        indicators = species_info.get('indicators', '未明确')
                        species_details.append("{}: {} ({})".format(species_name, change, indicators))
            species_str = '; '.join(species_details) if species_details else ''
            
            # 格式化多样性指标
            diversity_metrics = microbiome_analysis.get('diversity_metrics')
            diversity_str = str(diversity_metrics) if diversity_metrics else ''
            
            # 格式化整体模式
            overall_pattern = microbiome_analysis.get('overall_pattern')
            pattern_str = str(overall_pattern) if overall_pattern else ''
            
            result = {
                'PMID': paper['pmid'],
                '标题': paper['title'],
                '药物名称': analysis.get('drug_name'),
                '药物结构': analysis.get('drug_structure'),
                '副作用类型': '; '.join(side_effects) if side_effects else '',
                '菌群组学数据获取方法': analysis.get('microbiome_data_method'),
                '数据下载地址': analysis.get('data_download_url') or '; '.join(["{}: {}".format(k, v) for k, v in repo_info.items()]) if repo_info else '',
                '数据可用性声明': data_availability or '',
                '主要菌群门类': main_phyla_str,
                '具体菌种变化': species_str,
                '多样性指标': diversity_str,
                '菌群变化模式': pattern_str,
                '置信度': analysis.get('confidence_score', 0),
                '期刊': paper['journal'],
                '发表年份': paper['year'],
                '作者': paper['authors'],
                'PubMed链接': paper['url']
            }
            
            analyzed_results.append(result)
            time.sleep(1)  # 避免API速率限制
        
        self.results = analyzed_results
        return analyzed_results
    
    def save_to_excel(self, filename='药物副作用_菌群组学文献分析.xlsx'):
        if not self.results:
            print("没有结果可保存")
            return
        
        df = pd.DataFrame(self.results)
        
        column_order = [
            'PMID', '标题', '药物名称', '药物结构', '副作用类型',
            '菌群组学数据获取方法', '数据下载地址', '数据可用性声明',
            '主要菌群门类', '具体菌种变化', '多样性指标', '菌群变化模式',
            '置信度', '期刊', '发表年份', '作者', 'PubMed链接'
        ]
        
        df = df[column_order]
        
        try:
            df.to_excel(filename, index=False, engine='openpyxl')
            print("结果已保存到 {}".format(filename))
            print("共分析了 {} 篇文献".format(len(self.results)))
        except Exception as e:
            print("保存Excel文件时出错: {}".format(e))
            # 尝试保存为CSV
            csv_filename = filename.replace('.xlsx', '.csv')
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            print("结果已保存到 {}".format(csv_filename))
    
    def save_to_csv(self, filename='药物副作用_菌群组学文献分析.csv'):
        if not self.results:
            print("没有结果可保存")
            return
        
        df = pd.DataFrame(self.results)
        
        column_order = [
            'PMID', '标题', '药物名称', '药物结构', '副作用类型',
            '菌群组学数据获取方法', '数据下载地址', '数据可用性声明',
            '主要菌群门类', '具体菌种变化', '多样性指标', '菌群变化模式',
            '置信度', '期刊', '发表年份', '作者', 'PubMed链接'
        ]
        
        df = df[column_order]
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print("结果已保存到 {}".format(filename))

def main():
    print("=== 药物副作用与菌群组学文献分析工具 ===")
    print()
    
    # 使用配置文件中的API密钥，如果配置为空则询问用户
    pubmed_api_key = PUBMED_API_KEY if PUBMED_API_KEY else input("请输入PubMed API密钥（可选，直接回车跳过）: ").strip()
    ai_api_key = DEEPSEEK_API_KEY if DEEPSEEK_API_KEY else input("请输入DeepSeek API密钥（可选，直接回车跳过，将使用基础分析）: ").strip()
    
    print()
    
    # 创建分析器实例
    analyzer = PubMedLiteratureAnalyzer(
        api_key=pubmed_api_key if pubmed_api_key else None,
        ai_api_key=ai_api_key if ai_api_key else None
    )
    
    # 设置搜索关键词 - 使用专业的PubMed搜索语法
    # 药物结构维度（使用更宽泛的词以提高覆盖面）
    structure_part = (
        '(structure[tiab] OR "chemical structure"[tiab]'
        ' OR "molecular structure"[tiab] OR "structural formula"[tiab]'
        ' OR "drug structure"[tiab]'
        ' OR "Molecular Structure"[Mesh] OR "Chemistry, Pharmaceutical"[Mesh])'
    )
    
    # 副作用维度（25大类，用 OR 合并，兼顾自由词和 MeSH）
    side_effect_part = (
        # 1. 胃肠道
        '(diarrhea[tiab] OR colitis[tiab] OR gastrointestinal toxicity[tiab]'
        ' OR "Gastrointestinal Diseases"[Mesh]'
        # 2. 肝脏
        ' OR hepatotoxicity[tiab] OR "drug-induced liver injury"[tiab]'
        ' OR "Chemical and Drug Induced Liver Injury"[Mesh]'
        # 3. 心脏
        ' OR cardiotoxicity[tiab] OR "drug-induced cardiac toxicity"[tiab]'
        ' OR "Cardiotoxicity"[Mesh] OR "Long QT Syndrome"[Mesh]'
        # 4. 肾脏
        ' OR nephrotoxicity[tiab] OR "acute kidney injury"[tiab]'
        ' OR "Acute Kidney Injury"[Mesh]'
        # 5. 代谢/内分泌
        ' OR hyperglycemia[tiab] OR "metabolic side effect"[tiab] OR "weight gain"[tiab]'
        ' OR "Diabetes Mellitus, Type 2"[Mesh]'
        # 6. 神经/精神
        ' OR neurotoxicity[tiab] OR "chemotherapy-induced peripheral neuropathy"[tiab]'
        ' OR "Neurotoxicity Syndromes"[Mesh]'
        # 7. 血液
        ' OR myelosuppression[tiab] OR neutropenia[tiab] OR thrombocytopenia[tiab]'
        ' OR "Neutropenia"[Mesh] OR "Anemia"[Mesh]'
        # 8. 皮肤
        ' OR "skin toxicity"[tiab] OR rash[tiab] OR "hand-foot syndrome"[tiab]'
        ' OR "Drug Eruptions"[Mesh] OR "Alopecia"[Mesh]'
        # 9. 免疫相关不良反应
        ' OR "immune-related adverse event"[tiab] OR irAE[tiab]'
        ' OR "immune checkpoint inhibitor toxicity"[tiab]'
        ' OR "Cytokine Release Syndrome"[Mesh]'
        # 10. 呼吸/肺部
        ' OR "pulmonary toxicity"[tiab] OR "interstitial lung disease"[tiab]'
        ' OR "pulmonary fibrosis"[tiab]'
        ' OR "Lung Diseases, Interstitial"[Mesh]'
        # 11. 骨骼肌肉
        ' OR myalgia[tiab] OR arthralgia[tiab] OR rhabdomyolysis[tiab]'
        ' OR "bone toxicity"[tiab] OR osteoporosis[tiab]'
        ' OR "Muscular Diseases"[Mesh] OR "Bone Diseases, Metabolic"[Mesh]'
        # 12. 眼
        ' OR "ocular toxicity"[tiab] OR retinopathy[tiab] OR "vision disorder"[tiab]'
        ' OR "Eye Diseases"[Mesh]'
        # 13. 耳
        ' OR ototoxicity[tiab] OR "hearing loss"[tiab] OR tinnitus[tiab]'
        ' OR "Ear Diseases"[Mesh]'
        # 14. 生殖泌尿
        ' OR "reproductive toxicity"[tiab] OR infertility[tiab] OR "sexual dysfunction"[tiab]'
        ' OR "urinary retention"[tiab] OR "menstrual disorder"[tiab]'
        ' OR "Reproductive and Urinary Physiological Phenomena"[Mesh]'
        # 15. 血管/循环
        ' OR hypertension[tiab] OR thrombosis[tiab] OR edema[tiab]'
        ' OR "bleeding"[tiab] OR "vascular toxicity"[tiab]'
        ' OR "Vascular Diseases"[Mesh]'
        # 16. 疲劳/衰弱
        ' OR fatigue[tiab] OR asthenia[tiab]'
        ' OR "Fatigue"[Mesh] OR "Asthenia"[Mesh]'
        # 17. 过敏/超敏反应
        ' OR hypersensitivity[tiab] OR anaphylaxis[tiab]'
        ' OR "Drug Hypersensitivity"[Mesh]'
        # 18. 口腔黏膜/唾液腺
        ' OR "oral mucositis"[tiab] OR xerostomia[tiab]'
        ' OR "Stomatitis"[Mesh] OR "Xerostomia"[Mesh]'
        # 19. 味觉/嗅觉
        ' OR dysgeusia[tiab] OR anosmia[tiab]'
        ' OR "Taste Disorders"[Mesh] OR "Olfaction Disorders"[Mesh]'
        # 20. 药物诱发的感染
        ' OR "drug-induced infection"[tiab] OR "Clostridium difficile"[tiab]'
        ' OR "opportunistic infection"[tiab]'
        ' OR "Clostridium Infections"[Mesh]'
        # 21. 胰腺毒性
        ' OR "drug-induced pancreatitis"[tiab] OR pancreatitis[tiab]'
        ' OR "Pancreatitis"[Mesh]'
        # 22. 甲状腺功能异常
        ' OR "thyroid dysfunction"[tiab] OR thyroiditis[tiab]'
        ' OR "Thyroid Diseases"[Mesh]'
        # 23. 电解质紊乱
        ' OR "electrolyte imbalance"[tiab] OR hypokalemia[tiab]'
        ' OR "Water-Electrolyte Imbalance"[Mesh]'
        # 24. 认知障碍/脑雾
        ' OR "cognitive impairment"[tiab] OR "brain fog"[tiab]'
        ' OR "Cognitive Dysfunction"[Mesh]'
        # 25. 性腺/生殖内分泌紊乱
        ' OR "gonadal toxicity"[tiab] OR hypogonadism[tiab] OR hirsutism[tiab]'
        ' OR "Gonadal Disorders"[Mesh]'
        # 兜底泛化词
        ' OR "adverse effect"[tiab] OR "side effect"[tiab]'
        ' OR "Drug-Related Side Effects and Adverse Reactions"[Mesh])'
    )
    
    # 菌群测序维度
    microbiome_part = (
        '("microbiome sequencing"[tiab] OR "microbiota sequencing"[tiab]'
        ' OR "16S rRNA"[tiab] OR "metagenomic sequencing"[tiab]'
        ' OR "gut microbiota"[tiab]'
        ' OR "Microbiota"[Mesh] OR "Gastrointestinal Microbiome"[Mesh]'
        ' OR "Metagenomics"[Mesh])'
    )
    
    # 组合成最终查询
    final_query = "{} AND {} AND {}".format(structure_part, side_effect_part, microbiome_part)
    keywords = [final_query]
    
    print("搜索关键词:")
    print("=" * 80)
    print("结构部分:")
    print(structure_part)
    print()
    print("副作用部分（25大类）:")
    print(side_effect_part)
    print()
    print("微生物组部分:")
    print(microbiome_part)
    print()
    print("完整查询:")
    print(final_query)
    print("=" * 80)
    print()
    
    # 设置要分析的文献数量
    try:
        max_results_input = input("请输入要分析的文献数量（默认{}篇）: ".format(DEFAULT_MAX_RESULTS)).strip()
        max_results = int(max_results_input) if max_results_input else DEFAULT_MAX_RESULTS
    except ValueError:
        max_results = DEFAULT_MAX_RESULTS
        print("使用默认值: {}篇".format(DEFAULT_MAX_RESULTS))
    
    print()
    print("开始分析...")
    print("=" * 50)
    
    # 处理文献
    results = analyzer.process_papers(keywords, max_results)
    
    if results:
        print()
        print("=" * 50)
        print("分析完成！")
        print()
        
        # 显示结果摘要
        print("结果摘要:")
        print("总文献数: {}".format(len(results)))
        
        drugs_found = sum(1 for r in results if r['药物名称'])
        print("找到药物信息的文献: {}".format(drugs_found))
        
        side_effects_found = sum(1 for r in results if r['副作用类型'])
        print("找到副作用信息的文献: {}".format(side_effects_found))
        
        data_methods_found = sum(1 for r in results if r['菌群组学数据获取方法'])
        print("找到数据获取方法的文献: {}".format(data_methods_found))
        
        download_urls_found = sum(1 for r in results if r['数据下载地址'])
        print("找到数据下载地址的文献: {}".format(download_urls_found))
        
        print()
        
        # 保存结果
        save_choice = input("是否保存结果到Excel文件？(y/n, 默认y): ").strip().lower()
        if save_choice != 'n':
            filename = input("请输入文件名（默认: {}）: ".format(DEFAULT_OUTPUT_FILE)).strip()
            if not filename:
                filename = DEFAULT_OUTPUT_FILE
            
            if not filename.endswith('.xlsx') and not filename.endswith('.csv'):
                filename += '.xlsx'
            
            if filename.endswith('.xlsx'):
                analyzer.save_to_excel(filename)
            else:
                analyzer.save_to_csv(filename)
        
        print()
        print("分析完成！")
    else:
        print("未找到相关文献或分析失败")

if __name__ == "__main__":
    main()
