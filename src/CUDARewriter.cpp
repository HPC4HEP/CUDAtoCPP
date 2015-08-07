/*
 * CUDARewriter.cpp
 *
 *  Created on: 24/jul/15
 *      Author: latzori
 */

#include <string>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <sstream>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"


#include "clang/AST/Decl.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PPCallbacks.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm::sys::path;

//TODO this is mandatory?
static llvm::cl::OptionCategory MatcherSampleCategory("Matcher Sample");

class MyASTConsumer : public ASTConsumer {
public:
	MyASTConsumer(CompilerInstance *comp, Rewriter *R) : ASTConsumer(), CI(comp), Rew(R){ }
	virtual ~MyASTConsumer() { }

	virtual void Initialize(ASTContext &Context) {
		SM = &Context.getSourceManager();
		LO = &CI->getLangOpts();
		PP = &CI->getPreprocessor();
		//PP->addPPCallbacks(new RewriteIncludesCallback(this));
	}

	virtual bool HandleTopLevelDecl(DeclGroupRef DG) {

		Decl *firstDecl = DG.isSingleDecl() ? DG.getSingleDecl() : DG.getDeclGroup()[0];
		//SourceLocation loc = firstDecl->getLocation();
		SourceLocation sloc = SM->getSpellingLoc(firstDecl->getLocation());

		//std::cout << "loc " << loc.printToString(*SM) << "\n";
		//std::cout << "sloc " << sloc.printToString(*SM) << "\n";

		//Don't use extern in the include files!!!
//		std::cout << "loc " << loc.printToString(*SM) <<" (filename: " << SM->getFilename(loc).str() << ")\n";
//		std::cout << "sloc " << sloc.printToString(*SM) <<" (filename: " << SM->getFilename(sloc).str() << ")\n";
		//SM->getFileID(loc) != SM->getMainFileID() &&
//		std::cout << "MAIN ID " << SM->getMainFileID().getHashValue();
//		std::cout << " our ID " << SM->getFileID(loc).getHashValue();
//		std::cout << " our s ID " << SM->getFileID(sloc).getHashValue() << "\n";

		// Checking which file we are scanning (same as isFromMainFile property, now deprecated???)
		if( SM->getFileID(sloc) != SM->getMainFileID()){
//			std::cout << "FileID mismatch (Skipped rewrite)\n";
			return true; //Just skip the loc (TODO skip the file?) but continue parsing
		}

		std::cout << "HandleTopLevelDecl: " << sloc.printToString(*SM) << "\n";

        //Walk declarations in group and rewrite
        for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
            if (DeclContext *dc = dyn_cast<DeclContext>(*i)) {
            	std::cout << "\tDeclContext dc\n";
                //Basically only handles C++ member functions
                for (DeclContext::decl_iterator di = dc->decls_begin(), de = dc->decls_end(); di != de; ++di) {
                    if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*di)) {
                    	std::cout << "\t\tFunctionDecl fd\n";
                        //prevent implicitly defined functions from being rewritten
                    	// (since there's no source to rewrite..)
                        if (!fd->isImplicit()) {
                        	std::cout << "\t\t\tfd->isImplicit = false\n";
                            RewriteHostFunction(fd);
//                            RemoveFunction(fd, KernelRewrite);
//                        	if (fd->getNameAsString() == MainFuncName) {
//                                RewriteMain(fd);
//                            }
                        } else {
                            std::cout << "\t\t\tfd->isImplicit = true (skip)\n";
                        }
                    }
                }
            }
            //Handles globally defined C or C++ functions
            if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*i)) {
            	std::cout << "\tFunctionDecl fd\n";
            	//TODO: Don't translate explicit template specializations
            	//fixme ignoring templates for now
//                if(fd->getTemplatedKind() == clang::FunctionDecl::TK_NonTemplate || fd->getTemplatedKind() == FunctionDecl::TK_FunctionTemplate) {
                    if (fd->hasAttr<CUDAGlobalAttr>() || fd->hasAttr<CUDADeviceAttr>()) {
                    	std::cout << "\t\tfd->hasAttr<CUDAGlobalAttr>() || fd->hasAttr<CUDADeviceAttr>() = true\n";
                    	//Device function, so rewrite kernel
                        RewriteKernelFunction(fd);
                        if (fd->hasAttr<CUDAHostAttr>()){
                        	std::cout << "\t\t\tfd->hasAttr<CUDAHostAttr>() = true\n";
                            //Also a host function, so rewrite host
                            RewriteHostFunction(fd);
                        } else {
                        	std::cout << "\t\t\tfd->hasAttr<CUDAHostAttr>() = false\n";
                            //Simply a device function, so remove from host
//                            RemoveFunction(fd, HostRewrite);
                        }
                    } else {
                    	std::cout << "\t\tfd->hasAttr<CUDAGlobalAttr>() || fd->hasAttr<CUDADeviceAttr>() = false\n";
                        //Simply a host function, so rewrite
                        RewriteHostFunction(fd);
//                        if (CUDAHostAttr *attr = fd->getAttr<CUDAHostAttr>()) {
//                        	std::cout << "DEBUG: Found CUDAHostAttr...";
//                            SourceLocation instLoc = SM->getExpansionLoc(attr->getLocation());
//                            std::cout << " at loc " << instLoc.printToString(*SM);
//                            SourceRange realRange(instLoc, PP->getLocForEndOfToken(instLoc));
//                            std::cout << " ending at " << PP->getLocForEndOfToken(instLoc).printToString(*SM);
//                            Rew->ReplaceText(instLoc, Rew->getRangeSize(realRange), "");
//                            std::cout << " Removed!\n";
//                        }
                        //and remove from kernel
//                        RemoveFunction(fd, KernelRewrite);

//                        if (fd->getNameAsString() == MainFuncName) {
//                            RewriteMain(fd);
//                        }
                    }
//                } else {
//                    if (fd->getTemplateSpecializationInfo())
//                    	std::cout << "DEBUG: fd->getTemplateSpecializationInfo = true (Skip?)\n";
//                    else
//                    	std::cout << "DEBUG: Non-rewriteable function without TemplateSpecializationInfo detected?\n";
//                }
            } else if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
            	std::cout << "\tVarDecl vd\n";
//                RemoveVar(vd, KernelRewrite);
                RewriteHostVarDecl(vd);
            //Rewrite Structs here
            //Ideally, we should keep the expression inside parentheses ie __align__(<keep this>)
            // and just wrap it with __attribute__((aligned (<kept Expr>)))
            //TODO: Finish struct attribute translation
        	} else if (RecordDecl * rd = dyn_cast<RecordDecl>(*i)) {
        		std::cout << "\tRecordDecl rd\n";
                if (rd->hasAttrs()) {
                	std::cout << "\t\trd->hasAttrs = true\n";
//                    for (Decl::attr_iterator at = rd->attr_begin(), at_e = rd->attr_end(); at != at_e; ++at) {
//                        if (AlignedAttr *align = dyn_cast<AlignedAttr>(*at)) {
//                            if (!align->isAlignmentDependent()) {
//                                llvm::errs() << "Found an aligned struct of size: " << align->getAlignment(rd->getASTContext()) << " (bits)\n";
//                            } else {
//                                llvm::errs() << "Found a dependent alignment expresssion\n";
//                            }
//                        } else {
//                            llvm::errs() << "Found other attrib\n";
//                        }
//                    }
                }
            }
            //TODO rewrite type declarations
        }
        std::cout <<"\n";
        return true;
}

private:
    CompilerInstance *CI;
    SourceManager *SM;
    LangOptions *LO;
    Preprocessor *PP;
    Rewriter *Rew;

    //Simple function to strip attributes from host functions that may be declared as
    // both __host__ and __device__, then passes off to the host-side statement rewriter
    void RewriteHostFunction(FunctionDecl *hostFunc) {
		SourceLocation sloc = SM->getSpellingLoc(hostFunc->getLocation());
    	std::cout << "RewriteHostFunction hostFunc: " << sloc.printToString(*SM) << "\n";
    	//Remove any CUDA function attributes
        if (CUDAHostAttr *attr = hostFunc->getAttr<CUDAHostAttr>()) {
        	std::cout << "\thostFunc->hasAttr<CUDAHostAttr>() = true\n";
            RewriteAttr(attr, "", *Rew);
        }
        if (CUDADeviceAttr *attr = hostFunc->getAttr<CUDADeviceAttr>()) {
        	std::cout << "\thostFunc->hasAttr<CUDADeviceAttr>() = true\n";
            RewriteAttr(attr, "", *Rew);
        }

        //Rewrite the body
        if (Stmt *body = hostFunc->getBody()) {
        	std::cout << "\thostFunc->hasBody() = true\n";
            RewriteHostStmt(body);
        }
        //CurVarDeclGroups.clear();
    }

    void RewriteHostStmt(Stmt *s) {
    	std::cout << "RewriteHostStmt s: ";
        if (Expr *e = dyn_cast<Expr>(s)) {
        	SourceLocation sloc = SM->getSpellingLoc(e->getExprLoc());
        	std::cout << "<Expr> " << sloc.printToString(*SM) << "\n";
        	std::string str;
        	RewriteHostExpr(e, str);
        }
        else if (DeclStmt *ds = dyn_cast<DeclStmt>(s)) {
        	SourceLocation sloc = SM->getSpellingLoc(ds->getStartLoc());
        	std::cout << "<DeclStmt> " << sloc.printToString(*SM) << "\n";
        	DeclGroupRef DG = ds->getDeclGroup();
			Decl *firstDecl = DG.isSingleDecl() ? DG.getSingleDecl() : DG.getDeclGroup()[0];
            for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
                if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
                	std::cout << "\tVarDecl vd\n";
                    RewriteHostVarDecl(vd);
                }
            }
        } else {
            //Traverse children and recurse
            for (Stmt::child_iterator CI = s->child_begin(), CE = s->child_end(); CI != CE; ++CI) {
            	std::cout << "children!\n";
                if (*CI) RewriteHostStmt(*CI);
            }
        }
    }

    void RewriteKernelFunction(FunctionDecl* kf) {
		SourceLocation sloc = SM->getSpellingLoc(kf->getLocation());
    	std::cout << "RewriteKernelFunction kf: " << sloc.printToString(*SM) << "\n";
    	if (kf->hasAttr<CUDAGlobalAttr>()) {
    		std::cout << "\tkf->hasAttr<CUDAGlobalAttr>() = true\n";
        	//Means host callable
        }
        if (CUDAGlobalAttr *attr = kf->getAttr<CUDAGlobalAttr>()) {
    		std::cout << "\tkf->hasAttr<CUDAGlobalAttr>() = true\n";
    		RewriteAttr(attr, "", *Rew);
        }
        if (CUDADeviceAttr *attr = kf->getAttr<CUDADeviceAttr>()) {
    		std::cout << "\tkf->hasAttr<CUDADeviceAttr>() = true\n";
            RewriteAttr(attr, "", *Rew);
        }

        if (CUDAHostAttr *attr = kf->getAttr<CUDAHostAttr>()) {
    		std::cout << "\tkf->hasAttr<CUDAHostAttr>() = true\n";
            RewriteAttr(attr, "", *Rew);
        }

        //Rewrite formal parameters
        for (FunctionDecl::param_iterator PI = kf->param_begin(), PE = kf->param_end(); PI != PE; ++PI) {
        	std::cout << "\tParam! \n";
            RewriteKernelParam(*PI, kf->hasAttr<CUDAGlobalAttr>());
        }

        if (kf->hasBody()) {
        	std::cout << "\tkf->hasBody() = true\n";
            RewriteKernelStmt(kf->getBody());
        }


    }
    void RewriteKernelParam(ParmVarDecl *parmDecl, bool isFuncGlobal) {
		SourceLocation sloc = SM->getSpellingLoc(parmDecl->getLocation());
    	std::cout << "RewriteKernelParam parmDecl: " << sloc.printToString(*SM) << "\n";
    }

    void RewriteKernelStmt(Stmt *ks) {
    	std::cout << "RewriteKernelStmt ks: ";
    	 //Visit this node
		if (Expr *e = dyn_cast<Expr>(ks)) {
        	SourceLocation sloc = SM->getSpellingLoc(e->getExprLoc());
        	std::cout << "<Expr> " << sloc.printToString(*SM) << "\n";
			std::string str;
			if (RewriteKernelExpr(e, str)) {
				//ReplaceStmtWithText(e, str, KernelRewrite);
			}
		}
		else if (DeclStmt *ds = dyn_cast<DeclStmt>(ks)) {
        	SourceLocation sloc = SM->getSpellingLoc(ds->getStartLoc());
        	std::cout << "<DeclStmt> " << sloc.printToString(*SM) << "\n";
			DeclGroupRef DG = ds->getDeclGroup();
			for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
				if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
                	std::cout << "\tVarDecl vd\n";
					RewriteKernelVarDecl(vd);
				}
				//TODO other non-top level declarations??
			}
		}
		//TODO rewrite any other Stmts?

		else {
			//Traverse children and recurse
			for (Stmt::child_iterator CI = ks->child_begin(), CE = ks->child_end(); CI != CE; ++CI) {
            	std::cout << "children!\n";
				if (*CI) RewriteKernelStmt(*CI);
			}
		}
    }

    bool RewriteKernelExpr(Expr *e, std::string &newExpr) {
		SourceLocation sloc = SM->getSpellingLoc(e->getExprLoc());
    	std::cout << "RewriteKernelExpr e: " << sloc.printToString(*SM) << "\n";
    }

    void RewriteKernelVarDecl(VarDecl *var) {
		SourceLocation sloc = SM->getSpellingLoc(var->getLocation());
    	std::cout << "RewriteKernelVarDecl var: " << sloc.printToString(*SM) << "\n";
        if (CUDASharedAttr *sharedAttr = var->getAttr<CUDASharedAttr>()) {
        	std::cout << "\tvar->hasAttr<CUDASharedAttr>() = true\n";
            RewriteAttr(sharedAttr, "", *Rew);
            if (CUDADeviceAttr *devAttr = var->getAttr<CUDADeviceAttr>()){
            	std::cout << "\t\tvar->hasAttr<CUDADeviceAttr>() = true\n";
            	RewriteAttr(devAttr, "", *Rew);
            }
            //TODO rewrite extern shared mem
            //if (var->isExtern())?
        }

    }

    void RewriteHostVarDecl(VarDecl* var){
		SourceLocation sloc = SM->getSpellingLoc(var->getLocation());
    	std::cout << "RewriteHostVarDecl var: " << sloc.printToString(*SM) << "\n";
    	if (CUDAConstantAttr *constAttr = var->getAttr<CUDAConstantAttr>()) {
        	std::cout << "\tvar->hasAttr<CUDAConstantAttr>() = true\n";
    		//TODO: Do something with __constant__ memory declarations
            RewriteAttr(constAttr, "", *Rew);
            if (CUDADeviceAttr *devAttr = var->getAttr<CUDADeviceAttr>()){
            	std::cout << "\t\tvar->hasAttr<CUDADeviceAttr>() = true\n";
            	RewriteAttr(devAttr, "", *Rew);
            }
//            //DeviceMemVars.insert(var);
//            ConstMemVars.insert(var);
//
//            TypeLoc origTL = var->getTypeSourceInfo()->getTypeLoc();
//            if (LastLoc.isNull() || origTL.getBeginLoc() != LastLoc.getBeginLoc()) {
//                LastLoc = origTL;
//                RewriteType(origTL, "cl_mem", HostRewrite);
//            }
//            return;
        }
        else if (CUDASharedAttr *sharedAttr = var->getAttr<CUDASharedAttr>()) {
        	std::cout << "\tvar->hasAttr<CUDASharedAttr>() = true\n";
            //Handle __shared__ memory declarations
            RewriteAttr(sharedAttr, "", *Rew);
            if (CUDADeviceAttr *devAttr = var->getAttr<CUDADeviceAttr>()){
            	std::cout << "\t\ttvar->hasAttr<CUDADeviceAttr>() = true\n";
            	RewriteAttr(devAttr, "", *Rew);
            }
//            //TODO rewrite shared mem
//            //If extern, remove extern keyword and make into pointer
//            //if (var->isExtern())
//            SharedMemVars.insert(var);
        }
        else if (CUDADeviceAttr *attr = var->getAttr<CUDADeviceAttr>()) {
        	std::cout << "\tvar->hasAttr<CUDADeviceAttr>() = true\n";
            //Handle __device__ memory declarations
            RewriteAttr(attr, "", *Rew);
            //TODO add to devmems, rewrite type
        }
    }

    bool RewriteHostExpr(Expr *e, std::string &newExpr) {
		SourceLocation sloc = SM->getSpellingLoc(e->getExprLoc());
    	std::cout << "RewriteHostExpr e: " << sloc.printToString(*SM) << "\n";
        if (clang::CUDAKernelCallExpr *kce = dyn_cast<clang::CUDAKernelCallExpr>(e)) {
        	std::cout << "\tCUDAKernelCallExpr kce\n";
        	newExpr = RewriteCUDAKernelCall(kce);
        	return true;
        } else if  (CallExpr *ce = dyn_cast<CallExpr>(e)) {
        	if (ce->getDirectCallee()->getNameAsString().find("cuda") == 0)
        		std::cout << "\tCallExpr ce "<< ce->getDirectCallee()->getNameAsString() << "\n";
        		return RewriteCUDACall(ce, newExpr);
        } else if (MemberExpr *me = dyn_cast<MemberExpr>(e)) {
        	std::cout << "\tMemberExpr me\n";
        	//Catches expressions which refer to the member of a struct or class
        	// in the CUDA case these are primarily just dim3s and cudaDeviceProp
        } else if (ExplicitCastExpr *ece = dyn_cast<ExplicitCastExpr>(e)) {
        	std::cout << "\tExplicitCastExpr ece\n";
        	//Rewrite explicit casts of CUDA data types
        } else if (UnaryExprOrTypeTraitExpr *soe = dyn_cast<UnaryExprOrTypeTraitExpr>(e)) {
        	std::cout << "\tUnaryExprOrTypeTraitExpr soe\n";
        	//Rewrite unary expressions or type trait expressions (things like sizeof)
        } else if (CXXTemporaryObjectExpr *cte = dyn_cast<CXXTemporaryObjectExpr>(e)) {
        	std::cout << "\tCXXTemporaryObjectExpr cte\n";
        	//Catches dim3 declarations of the form: some_var=dim3(x,y,z);
        	// the RHS is considered a temporary object
        } else if (CXXConstructExpr *cce = dyn_cast<CXXConstructExpr>(e)) {
        	std::cout << "\tCXXConstructExpr cce\n";
        	//Catches dim3 declarations of the form: dim3 some_var(x,y,z);
        }
        bool ret = false;
        //Do a DFS, recursing into children, then rewriting this expression
        //if rewrite happened, replace text at old sourcerange
        for (Stmt::child_iterator CI = e->child_begin(), CE = e->child_end(); CI != CE; ++CI) {
            std::string s;
            Expr *child = (Expr *) *CI;
            if (child && RewriteHostExpr(child, s)) {
            	std::cout << "I was a child?\n";
                //Perform "rewrite", which is just a simple replace
                //ReplaceStmtWithText(child, s, exprRewriter);
                ret = true;
            }
        }


//        SourceRange newrealRange(SM->getExpansionLoc(e->getLocStart()),
//                              SM->getExpansionLoc(e->getLocEnd()));
//        newExpr = exprRewriter.getRewrittenText(realRange);
        return ret;
    }

    std::string RewriteCUDAKernelCall(clang::CUDAKernelCallExpr *kernelCall) {
		SourceLocation sloc = SM->getSpellingLoc(kernelCall->getExprLoc());
    	std::cout << "RewriteCUDAKernelCall kernelCall: " << sloc.printToString(*SM) << "\n";
        CallExpr *kernelConfig = kernelCall->getConfig();
        Expr *grid = kernelConfig->getArg(0);
        Expr *block = kernelConfig->getArg(1);
        return "";
    }

    bool RewriteCUDACall(CallExpr *cudaCall, std::string &newExpr) {
		SourceLocation sloc = SM->getSpellingLoc(cudaCall->getExprLoc());
    	std::cout << "RewriteCUDACall cudaCall: " << sloc.printToString(*SM) << "\n";
        std::string funcName = cudaCall->getDirectCallee()->getNameAsString();
        //TODO we have to match funcName.
        return true;
    }



    //The workhorse that takes the constructed replacement attribute and inserts it in place of the old one
    void RewriteAttr(Attr *attr, std::string replace, Rewriter &rewrite){
    	std::cout << "RewriteAttr!\n";
        SourceLocation instLoc = SM->getExpansionLoc(attr->getLocation());
        SourceRange realRange(instLoc, PP->getLocForEndOfToken(instLoc));
        rewrite.ReplaceText(instLoc, rewrite.getRangeSize(realRange), replace);
    }
};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {}
  void EndSourceFileAction() override {
    TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(llvm::outs());
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<MyASTConsumer>(&CI, &TheRewriter);
  }

private:
  Rewriter TheRewriter;
};

int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, MatcherSampleCategory);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());

  return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
